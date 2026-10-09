//! The shared chat store's read path against goldens recorded from the real
//! Python (the chat core's spec row C2; the native chat's spec §5.2).
//!
//! `tools/lattice_native_parity.py` runs `history.py` and writes what it
//! decides to `tests/parity/chat/`:
//! - `store_parse.json`: `_parse_turn` over turn lines, `load_thread` over
//!   thread files, `_thread_path` over ids;
//! - `store_index.json`: `_read_index_locked` over index files;
//! - `titles.json`: `auto_title` and `rename`'s normalisation;
//! - `evicted_read.json`: the archive's reader, `_read_evicted_index_locked`,
//!   over archive index files (row G1; it replaced C2's provisional fixtures,
//!   which ran the web privacy fix's commit before it was merged).
//!
//! The write path's goldens (`store_ops.json`, `evicted_ops.json`,
//! `archive_row.json`) are replayed by the store's own unit tests
//! (`chat::write_parity_tests`, row G2), the only place the write gate can
//! be opened.
//!
//! Every number Python produced is recorded as text (`repr`, `str`) and
//! compared as text. A case Python reads and this port cannot hold is tagged
//! `deviation` by the case's author (D1 for a turn, D7 for an index): this
//! test holds the port to refusing it, as a skipped line or an index that is
//! left alone, never to guessing at it.
//!
//! Files are written only in temporary folders this test makes.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lattice_agents::model::InputItem;
use lattice_core::calc::float_repr;
use lattice_core::chat::archive::{self, ArchiveState};
use lattice_core::chat::pyjson::{self, CoerceError, JsonError};
use lattice_core::chat::store::{
    self, IdSource, IndexState, MAX_THREADS, MAX_TURNS, SharedThreadStore, StoredTurn,
};
use lattice_core::chat::{echo, grounding, prompt, think};
use serde_json::Value;

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "lattice-core-chat-parity-{tag}-{}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn read(relative: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join(relative);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The generator's id source: `id` and ten digits, counted from 1.
fn counter() -> IdSource {
    let next = Arc::new(AtomicU64::new(1));
    Arc::new(move || format!("id{:010}", next.fetch_add(1, Ordering::Relaxed)))
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap())
        .collect()
}

/// Put a case's input at `path`, as the generator did.
fn materialise(spec: &Value, path: &Path) {
    if spec.get("missing").is_some() {
        return;
    }
    if spec.get("directory").is_some() {
        fs::create_dir_all(path).unwrap();
        return;
    }
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let bytes = match spec.get("hex").and_then(Value::as_str) {
        Some(digits) => hex(digits),
        None => spec["text"].as_str().unwrap().as_bytes().to_vec(),
    };
    fs::write(path, bytes).unwrap();
}

fn strs(value: &Value) -> Vec<&str> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_str().unwrap())
        .collect()
}

/// Deviation D2: where Python's `str()` of a list or a dict gives its repr, the
/// golden's `d2` map gives the JSON text this port writes instead.
fn same(native: &str, golden: &str, d2: &Value) -> bool {
    native == golden || d2.get(golden).and_then(Value::as_str) == Some(native)
}

fn same_list(native: &[String], golden: &Value, d2: &Value) -> bool {
    let golden = strs(golden);
    native.len() == golden.len() && native.iter().zip(golden).all(|(n, g)| same(n, g, d2))
}

fn assert_turn(turn: &StoredTurn, golden: &Value, d2: &Value, context: &str) {
    let text = |key: &str| golden[key].as_str().unwrap_or_else(|| panic!("{key}"));
    let count = |key: &str| golden[key].as_str().map(str::to_owned);
    for (key, native) in [
        ("id", &turn.id),
        ("role", &turn.role),
        ("text", &turn.text),
        ("provider", &turn.provider),
        ("error", &turn.error),
    ] {
        assert!(
            same(native, text(key), d2),
            "{key}: {native:?} against {:?}: {context}",
            text(key)
        );
    }
    assert_eq!(float_repr(turn.ts), text("ts"), "ts: {context}");
    assert!(
        same_list(&turn.tools, &golden["tools"], d2),
        "tools: {:?}: {context}",
        turn.tools
    );
    assert!(
        same_list(&turn.unsupported, &golden["unsupported"], d2),
        "unsupported: {:?}: {context}",
        turn.unsupported
    );
    assert_eq!(
        turn.facts.iter().map(pyjson::dumps).collect::<Vec<_>>(),
        strs(&golden["facts"]),
        "facts: {context}"
    );
    for (flag, value) in [
        ("constrained", turn.constrained),
        ("truncated", turn.truncated),
        ("cancelled", turn.cancelled),
        ("superseded", turn.superseded),
    ] {
        assert_eq!(Some(value), golden[flag].as_bool(), "{flag}: {context}");
    }
    assert_eq!(
        turn.prompt_tokens.map(|n| n.to_string()),
        count("prompt_tokens"),
        "prompt_tokens: {context}"
    );
    assert_eq!(
        turn.completion_tokens.map(|n| n.to_string()),
        count("completion_tokens"),
        "completion_tokens: {context}"
    );
}

#[test]
fn turn_lines_coerce_as_python_coerces_them() {
    let golden = read("chat/store_parse.json");
    let cases = golden["parse_turn"].as_array().unwrap();
    let (mut kept, mut skipped, mut refused) = (0, 0, 0);
    for case in cases {
        let line = case["line"].as_str().unwrap();
        let ids = counter();
        let parsed = pyjson::loads(line);
        if case.get("deviation").is_some() {
            // Python keeps a turn here; this port must not guess at it.
            match parsed {
                Err(JsonError::BeyondNative(_)) => {}
                Ok(record) => assert_eq!(
                    store::parse_turn(&record, &ids).err(),
                    Some(CoerceError::Native),
                    "{line}"
                ),
                Err(JsonError::Invalid) => panic!("Python read {line}"),
            }
            refused += 1;
            continue;
        }
        let record = parsed.unwrap_or_else(|e| panic!("{line}: {e:?}"));
        let result = store::parse_turn(&record, &ids);
        if case["turn"].is_null() {
            assert_eq!(result.err(), Some(CoerceError::Python), "{line}");
            skipped += 1;
        } else {
            assert_turn(&result.unwrap(), &case["turn"], &case["d2"], line);
            kept += 1;
        }
    }
    println!("parse_turn: {kept} kept, {skipped} skipped as Python skips, {refused} D1 refused");
    assert!(kept >= 100 && skipped >= 10 && refused >= 6);
}

#[test]
fn thread_files_load_as_python_loads_them() {
    let golden = read("chat/store_parse.json");
    assert_eq!(golden["max_turns"].as_u64(), Some(MAX_TURNS as u64));
    let cases = golden["load_thread"].as_array().unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let scratch = Scratch::new("load");
        materialise(&case["input"], &scratch.path().join("thread.jsonl"));
        let store = SharedThreadStore::with_ids(scratch.path(), counter());
        let turns = store.load("thread");
        if let Some(ids) = case.get("ids") {
            assert_eq!(
                turns.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(),
                strs(ids),
                "{name}"
            );
            continue;
        }
        let skips: Vec<&str> = case.get("native_skips").map(strs).unwrap_or_default();
        let expected: Vec<&Value> = case["turns"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|turn| !skips.contains(&turn["id"].as_str().unwrap()))
            .collect();
        assert_eq!(turns.len(), expected.len(), "{name}");
        for (turn, golden_turn) in turns.iter().zip(expected) {
            assert_turn(turn, golden_turn, &Value::Null, name);
        }
        // Reading wrote nothing.
        let names: Vec<_> = fs::read_dir(scratch.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert!(names.len() <= 1, "{name}: {names:?}");
    }
    println!("load_thread: {} files", cases.len());
}

#[test]
fn thread_ids_name_the_files_python_names() {
    let golden = read("chat/store_parse.json");
    for case in golden["thread_path"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        assert_eq!(
            store::thread_file_name(id).as_deref(),
            case["file"].as_str(),
            "{id:?}"
        );
    }
}

#[test]
fn the_index_reads_as_python_reads_it_or_is_left_alone() {
    let golden = read("chat/store_index.json");
    assert_eq!(golden["max_threads"].as_u64(), Some(MAX_THREADS as u64));
    let cases = golden["cases"].as_array().unwrap();
    let mut rows_seen = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let scratch = Scratch::new("index");
        materialise(&case["input"], &scratch.path().join(store::INDEX));
        let state = SharedThreadStore::new(scratch.path()).list();
        let python = case["python"].as_str().unwrap();
        if case.get("deviation").is_some() {
            assert_eq!(python, "parses", "{name}");
            assert!(case["python_row_count"].as_u64().unwrap() > 0, "{name}");
            assert_eq!(state, IndexState::Unreadable, "{name}: D7");
            continue;
        }
        match python {
            "missing" => assert_eq!(state, IndexState::Absent, "{name}"),
            "fails" => assert_eq!(state, IndexState::Unreadable, "{name}"),
            "parses" => {
                let IndexState::Rows(rows) = state else {
                    panic!("{name}: {state:?}")
                };
                let expected = case["rows"].as_array().unwrap();
                assert_eq!(rows.len(), expected.len(), "{name}");
                for (row, want) in rows.iter().zip(expected) {
                    let text = |key: &str| want[key].as_str().unwrap();
                    let d2 = &case["d2"];
                    assert!(same(&row.id, text("id"), d2), "{name}: {:?}", row.id);
                    assert!(
                        same(&row.title, text("title"), d2),
                        "{name}: {:?}",
                        row.title
                    );
                    assert_eq!(float_repr(row.created), text("created"), "{name}");
                    assert_eq!(float_repr(row.updated), text("updated"), "{name}");
                    assert_eq!(row.turns.to_string(), text("turns"), "{name}");
                    assert!(
                        same(&row.pinned_provider, text("pinned_provider"), d2),
                        "{name}"
                    );
                    rows_seen += 1;
                }
            }
            other => panic!("{name}: {other}"),
        }
    }
    println!("store_index: {} cases, {rows_seen} rows", cases.len());
}

#[test]
fn titles_are_cut_and_collapsed_as_python_does() {
    let golden = read("chat/titles.json");
    for case in golden["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        assert_eq!(
            store::auto_title(text),
            case["auto_title"].as_str().unwrap(),
            "{text:?}"
        );
        assert_eq!(
            store::normalize_title(text),
            case["rename"].as_str().unwrap_or(""),
            "{text:?}"
        );
    }
}

#[test]
fn the_archive_index_reads_as_python_reads_it_or_is_left_alone() {
    let golden = read("chat/evicted_read.json");
    assert_eq!(
        golden["source"].as_str(),
        Some("Python oracle/assistant/history.py")
    );
    let cases = golden["cases"].as_array().unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let scratch = Scratch::new("evicted");
        materialise(
            &case["input"],
            &scratch.path().join(archive::EVICTED_DIR).join("index.json"),
        );
        let state = archive::read_evicted(scratch.path());
        let python = case["python"].as_str().unwrap();
        if case.get("deviation").is_some() {
            assert_eq!(python, "rows", "{name}");
            assert_eq!(state, ArchiveState::Unreadable, "{name}: D7, never corrupt");
            continue;
        }
        if case["input"].get("missing").is_some() {
            assert_eq!(state, ArchiveState::Absent, "{name}");
            continue;
        }
        match python {
            "unreadable" => assert_eq!(state, ArchiveState::Unreadable, "{name}"),
            "corrupt" => assert_eq!(state, ArchiveState::Corrupt, "{name}"),
            "rows" => {
                let ArchiveState::Rows(rows) = state else {
                    panic!("{name}: {state:?}")
                };
                let raws: Vec<String> = rows.iter().map(|row| pyjson::dumps(row.raw())).collect();
                assert_eq!(raws, strs(&case["rows"]), "{name}");
            }
            other => panic!("{name}: {other}"),
        }
    }
    println!("evicted_read: {} cases", cases.len());
}

/// A history turn as `recent_exchanges` gives it: only its role and text matter.
fn history_turn(role: &str, text: &str) -> StoredTurn {
    StoredTurn {
        id: "t".into(),
        ts: 1.0,
        role: role.into(),
        text: text.into(),
        tools: Vec::new(),
        facts: Vec::new(),
        unsupported: Vec::new(),
        provider: String::new(),
        error: String::new(),
        constrained: false,
        truncated: false,
        cancelled: false,
        prompt_tokens: None,
        completion_tokens: None,
        superseded: false,
    }
}

#[test]
fn a_plain_turn_sends_what_the_web_sends() {
    let golden = read("chat/messages.json");
    let cases = golden["cases"].as_array().unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let recent: Vec<StoredTurn> = case["recent"]
            .as_array()
            .unwrap()
            .iter()
            .map(|turn| {
                history_turn(
                    turn["role"].as_str().unwrap(),
                    turn["text"].as_str().unwrap(),
                )
            })
            .collect();
        let request = prompt::messages(case["question"].as_str().unwrap(), &recent);
        let mut sent = vec![("system", request.system.clone())];
        for item in &request.input {
            sent.push(match item {
                InputItem::User(text) => ("user", text.clone()),
                InputItem::Assistant {
                    text: Some(text),
                    tool_calls,
                } if tool_calls.is_empty() => ("assistant", text.clone()),
                other => panic!("{name}: {other:?}"),
            });
        }
        let expected: Vec<(&str, String)> = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|message| {
                (
                    message["role"].as_str().unwrap(),
                    message["content"].as_str().unwrap().to_owned(),
                )
            })
            .collect();
        assert_eq!(sent, expected, "{name}");
    }
    println!("messages: {} stores", cases.len());
}

/// `text` cut at code point offsets `cuts`.
fn pieces<'a>(text: &'a str, cuts: &[usize]) -> Vec<&'a str> {
    let offsets: Vec<usize> = text
        .char_indices()
        .map(|(at, _)| at)
        .chain(std::iter::once(text.len()))
        .collect();
    let mut bounds = vec![0];
    bounds.extend(cuts.iter().map(|cut| offsets[*cut]));
    bounds.push(text.len());
    bounds
        .windows(2)
        .map(|pair| &text[pair[0]..pair[1]])
        .collect()
}

fn stream(fragments: &[&str]) -> String {
    let mut filter = think::ThinkFilter::new();
    let mut out: String = fragments
        .iter()
        .map(|fragment| filter.feed(fragment))
        .collect();
    out.push_str(&filter.flush());
    out
}

#[test]
fn the_scratchpad_is_removed_as_the_web_removes_it_however_it_streams() {
    let golden = read("chat/think.json");
    let cases = golden["cases"].as_array().unwrap();
    let mut streams = 0;
    for case in cases {
        let text = case["text"].as_str().unwrap();
        assert_eq!(
            think::strip_think(text),
            case["stripped"].as_str().unwrap(),
            "{text:?}"
        );
        let streamed = case["streamed"].as_str().unwrap();
        let length = text.chars().count();
        for position in 0..=length {
            assert_eq!(
                stream(&pieces(text, &[position])),
                streamed,
                "{text:?} cut at {position}"
            );
            streams += 1;
        }
        for cuts in case["chunkings"].as_array().unwrap() {
            let cuts: Vec<usize> = cuts
                .as_array()
                .unwrap()
                .iter()
                .map(|cut| cut.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(
                stream(&pieces(text, &cuts)),
                streamed,
                "{text:?} cut at {cuts:?}"
            );
            streams += 1;
        }
        let each: Vec<usize> = (1..length).collect();
        assert_eq!(
            stream(&pieces(text, &each)),
            streamed,
            "{text:?} one by one"
        );
    }
    println!("think: {} texts, {streams} streamings", cases.len());
}

#[test]
fn figures_no_tool_backs_are_found_as_the_web_finds_them() {
    let golden = read("chat/grounding.json");
    let cases = golden["cases"].as_array().unwrap();
    let (mut mentions_seen, mut deviations) = (0, 0);
    for case in cases {
        let prose = case["prose"].as_str().unwrap();
        let question = case["question"].as_str().unwrap();
        let found = grounding::find_mentions(prose);
        if case.get("deviation").is_some() {
            // D5: a digit of another script is not a digit here; the scanner
            // never reports one as part of a figure.
            println!(
                "D5 {prose:?}: python {:?}, native {:?}",
                case["mentions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|m| m["text"].clone())
                    .collect::<Vec<_>>(),
                found.iter().map(|m| m.text.clone()).collect::<Vec<_>>()
            );
            for mention in &found {
                assert!(
                    !mention
                        .text
                        .chars()
                        .any(|c| c.is_numeric() && !c.is_ascii_digit()),
                    "{prose:?}: {mention:?}"
                );
            }
            deviations += 1;
            continue;
        }
        let expected = case["mentions"].as_array().unwrap();
        assert_eq!(found.len(), expected.len(), "{prose:?}: {found:?}");
        for (mention, want) in found.iter().zip(expected) {
            assert_eq!(mention.text, want["text"].as_str().unwrap(), "{prose:?}");
            assert_eq!(
                float_repr(mention.value),
                want["value"].as_str().unwrap(),
                "{prose:?}"
            );
            assert_eq!(
                Some(mention.decimals as u64),
                want["decimals"].as_u64(),
                "{prose:?}"
            );
            assert_eq!(mention.unit, want["unit"].as_str().unwrap(), "{prose:?}");
            assert_eq!(Some(mention.scaled), want["scaled"].as_bool(), "{prose:?}");
            assert_eq!(
                Some(mention.start as u64),
                want["start"].as_u64(),
                "{prose:?}"
            );
            assert_eq!(Some(mention.end as u64), want["end"].as_u64(), "{prose:?}");
            mentions_seen += 1;
        }
        assert_eq!(
            grounding::unsupported_without_facts(prose, question),
            strs(&case["unsupported"]),
            "{prose:?} / {question:?}"
        );
    }
    println!(
        "grounding: {} proses, {mentions_seen} mentions, {deviations} D5",
        cases.len()
    );
    assert!(cases.len() >= 200 && mentions_seen >= 200 && deviations >= 3);
}

#[test]
fn the_echo_says_what_the_web_s_echo_says() {
    let golden = read("chat/echo.json");
    for case in golden["cases"].as_array().unwrap() {
        let question = case["question"].as_str().unwrap();
        let text = echo::echo_text(question);
        assert_eq!(text, case["text"].as_str().unwrap(), "{question:?}");
        let streamed = echo::echo_pieces(&text);
        assert_eq!(
            Some(streamed.len() as u64),
            case["pieces"].as_u64(),
            "{question:?}"
        );
        assert_eq!(
            streamed
                .iter()
                .map(|piece| piece.chars().count() as u64)
                .max(),
            case["longest"].as_u64(),
            "{question:?}"
        );
        assert_eq!(streamed.concat(), text);
    }
}

/// Row C4: the chat's vocabulary as the web's Python decides it
/// (`chat/vocabulary.json`). The ids it routes, the thread ids it accepts
/// and its message bound are held exactly, except where Python's `$`
/// accepts one final newline (native refuses; the case says so). The
/// picker's entries share the labels, the echo's words and the remote
/// endpoint's words; Auto's, Local's and Cloud's details, and a named
/// loopback endpoint's, are native by design (LR1: Local is the managed
/// server; a named endpoint is never Local) and are held to differ.
#[test]
fn the_vocabulary_is_the_webs() {
    use lattice_core::chat::vocab::{self, words};
    use lattice_core::env::MapEnv;
    use lattice_core::state::StateRoot;
    use lattice_protocol::chat::is_thread_id;

    let golden = read("chat/vocabulary.json");
    assert_eq!(
        golden["max_message_chars"].as_u64(),
        Some(vocab::MAX_MESSAGE_CHARS as u64)
    );
    let mut deviations = 0;
    for case in golden["valid_provider"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let dev_echo = case["dev_echo"].as_bool().unwrap();
        let python = case["valid"].as_bool().unwrap();
        let native = vocab::is_valid_choice(id, dev_echo);
        if case.get("deviation").is_some() {
            deviations += 1;
            assert!(python && !native && id.ends_with('\n'), "{id:?}");
        } else {
            assert_eq!(native, python, "{id:?} dev_echo={dev_echo}");
        }
    }
    for case in golden["thread_id"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let python = case["valid"].as_bool().unwrap();
        if case.get("deviation").is_some() {
            deviations += 1;
            assert!(python && !is_thread_id(id) && id.ends_with('\n'), "{id:?}");
        } else {
            assert_eq!(is_thread_id(id), python, "{id:?}");
        }
    }
    assert!(deviations >= 3, "the trailing-newline cases are recorded");

    let entries = golden["model_choices"].as_array().unwrap();
    let web = |id: &str| {
        entries
            .iter()
            .find(|entry| entry["id"] == id)
            .unwrap_or_else(|| panic!("the web lists {id}"))
    };
    for (id, label) in [
        ("auto", vocab::AUTO_LABEL),
        ("local", vocab::LOCAL_LABEL),
        ("cloud", vocab::CLOUD_LABEL),
        ("dev:echo", vocab::DEV_ECHO_LABEL),
    ] {
        assert_eq!(web(id)["label"], label, "{id}");
    }
    assert_eq!(web("dev:echo")["detail"], words::ECHO_DETAIL);
    assert_eq!(
        web("endpoint:hosted")["detail"],
        words::REMOTE_ENDPOINT_DETAIL
    );
    for (id, native) in [
        ("auto", words::AUTO_DETAIL),
        ("local", words::LOCAL_DETAIL),
        ("cloud", words::CLOUD_DETAIL),
    ] {
        assert_ne!(
            web(id)["detail"],
            native,
            "{id}: native's own words, by design"
        );
    }
    assert_eq!(
        web("endpoint:near")["locality"],
        "local",
        "the web calls a loopback endpoint local"
    );
    // Since the web's privacy fix, the web's Auto
    // and Local never leave the machine, so it calls both local, as native
    // always has (LR1).
    for id in ["auto", "local"] {
        assert_eq!(web(id)["locality"], "local", "{id}");
    }

    // The web lists its managed llama.cpp row. Native lists the same
    // row (the core's own server, Local by construction) with the web's
    // label, locality and words, and the endpoints in the web's order, over
    // the generator's fixture registry.
    let scratch = Scratch::new("choices");
    let state = StateRoot::at(scratch.path().join("root"));
    fs::create_dir_all(&state.globals).unwrap();
    fs::write(
        state.globals.join("model_endpoints.json"),
        r#"{"version": 1, "endpoints": [{"id": "near", "label": "Near server", "base_url": "http://127.0.0.1:8000/v1", "model": "m", "enabled": true}, {"id": "hosted", "label": "Hosted", "base_url": "https://hosted.example.test/v1", "model": "m", "api_key_name": "HOSTED_API_KEY", "enabled": true}]}"#,
    )
    .unwrap();
    let home = scratch.path().join("home");
    let env = Arc::new(
        MapEnv::new()
            .with("HOSTED_API_KEY", "fixture-hosted")
            .with("USERPROFILE", home.as_os_str())
            .with("HOME", home.as_os_str()),
    );
    let keys = lattice_core::keys::KeyStore::new(env.clone(), &state);
    let local = vocab::LocalView::read(env.as_ref(), &state);
    let native = vocab::chat_choices(env.as_ref(), &state, &keys, true, &local);
    for id in ["auto", "local"] {
        let entry = native
            .entries
            .iter()
            .find(|entry| entry.choice.id == id)
            .unwrap_or_else(|| panic!("native lists {id}"));
        assert_eq!(
            entry.choice.locality,
            lattice_protocol::Locality::Local,
            "{id}"
        );
    }
    let managed = native
        .entries
        .iter()
        .find(|entry| entry.choice.id == "endpoint:llamacpp-local")
        .expect("native lists the managed row");
    let web_managed = web("endpoint:llamacpp-local");
    assert_eq!(web_managed["label"], managed.choice.label.as_str());
    assert_eq!(web_managed["detail"], managed.choice.detail.as_str());
    assert_eq!(web_managed["detail"], words::MANAGED_ROW_DETAIL);
    assert_eq!(web_managed["locality"], "local");
    assert_eq!(managed.choice.locality, lattice_protocol::Locality::Local);
    let web_endpoints: Vec<&str> = entries
        .iter()
        .filter_map(|entry| entry["id"].as_str())
        .filter(|id| id.starts_with(vocab::ENDPOINT_PREFIX))
        .collect();
    let native_endpoints: Vec<&str> = native
        .entries
        .iter()
        .map(|entry| entry.choice.id.as_str())
        .filter(|id| id.starts_with(vocab::ENDPOINT_PREFIX))
        .collect();
    assert_eq!(native_endpoints, web_endpoints);
    drop(scratch);

    let scratch = Scratch::new("hf");
    let state = StateRoot::at(scratch.path());
    fs::create_dir_all(&state.globals).unwrap();
    let pref = state.globals.join("hf_model_dir.json");
    let cases = golden["hf_model_dir"].as_array().unwrap();
    assert!(cases.len() >= 20);
    for case in cases {
        let mut env = MapEnv::new();
        for (name, value) in case["env"].as_object().unwrap() {
            env.set(name, value.as_str().unwrap());
        }
        let _ = fs::remove_file(&pref);
        if let Some(text) = case["file"].as_str() {
            fs::write(&pref, hex(text)).unwrap();
        }
        assert_eq!(
            vocab::hf_model_configured(&env, &state),
            case["configured"].as_bool().unwrap(),
            "{case}"
        );
    }
}
