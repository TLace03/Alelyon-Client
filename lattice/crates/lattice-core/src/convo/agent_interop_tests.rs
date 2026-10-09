//! I14's sidecar half and I15 at the agent chat's level (spec §16.3; row
//! G3 ran them at the store seam only): `AgentChat` over the shared store
//! with its writes open through the `#[cfg(test)]` gate parameter (so they
//! do not depend on the production gate), against Python's `history.py` in
//! the repository's interpreter through `tools/lattice_chat_interop.py`, as G3 does.

use std::sync::Arc;
use std::time::Duration;

use lattice_agents::model::InputItem;
use lattice_agents::testing::ScriptedModel;
use lattice_protocol::conversation::{
    AgentChatService, ConversationEventKind, Decision, Mode, RegenerateRequest,
};
use serde_json::{Value, json};

use super::agent_tests::{H, call, local, say, user_texts};
use crate::chat::interop_tests::{Py, assert_archive_whole, int, listed_ids, python_load, text};
use crate::chat::store::{SharedThreadStore, uuid_ids};
use crate::chat::transcript::TranscriptStore;
use crate::testkit::TempDir;

/// A scripted model whose call number `held` (0-based) waits for the test's
/// word before it answers: a model that "goes on" for exactly as long as
/// the test needs, not for a fixed time.
struct Held {
    inner: Arc<ScriptedModel>,
    held: usize,
    calls: std::sync::atomic::AtomicUsize,
    gate: Arc<tokio::sync::Notify>,
}

impl lattice_agents::model::Model for Held {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn config_for_trace(&self) -> Value {
        self.inner.config_for_trace()
    }

    fn generation_trace(&self) -> lattice_agents::model::GenerationTrace {
        self.inner.generation_trace()
    }

    fn stream(
        &self,
        request: lattice_agents::model::ModelRequest,
    ) -> futures::stream::BoxStream<
        'static,
        Result<lattice_agents::model::ModelEvent, lattice_agents::model::ModelError>,
    > {
        use futures::{FutureExt, StreamExt};
        let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let answer = self.inner.stream(request);
        if n != self.held {
            return answer;
        }
        let gate = self.gate.clone();
        async move {
            gate.notified().await;
            answer
        }
        .flatten_stream()
        .boxed()
    }
}

fn python(name: &'static str) -> Option<Py> {
    match Py::new(name) {
        Ok(py) => Some(py),
        Err(why) => {
            println!("interop {name}: SKIPPED: {why}");
            None
        }
    }
}

fn harness(tag: &str, root: &std::path::Path) -> (H, Arc<SharedThreadStore>) {
    let store = Arc::new(SharedThreadStore::with_gate(root, uuid_ids(), true));
    let given: Arc<dyn TranscriptStore> = store.clone();
    let h = H::with(tag, &[], move |config| config.store = given);
    (h, store)
}

/// I15 through the agent chat: a native agent turn waits on an approval
/// while Python creates threads past the cap; after the decision the answer
/// is saved and the thread is still listed (S25: `touch` at TurnStart,
/// after the decision and before `append_answer`).
/// Mutant: `touch` never called (the answer is lost).
#[test]
fn i15_agent_a_turn_waiting_on_an_approval_is_not_archived() {
    let Some(py) = python("I15-agent") else {
        return;
    };
    let dir = TempDir::new("agent-i15");
    let root = dir.path().join("store");
    let (h, store) = harness("agent-i15", &root);
    py.run(
        &root,
        "fill-past-cap",
        &["--count", "59", "--prefix", "older"],
    );
    let ws = h.workspace();
    // The model's answer after the approval waits until the 60th thread is
    // made (the test's word, not a 3 s delay), so a slow Python under load
    // cannot let the answer land first and make the check vacuous.
    let gate = Arc::new(tokio::sync::Notify::new());
    h.answer_with(Arc::new(Held {
        inner: Arc::new(ScriptedModel::new(vec![
            call("run_command", json!({"command": "Write-Output hi"}), "c1"),
            say("the agent's answer"),
        ])),
        held: 1,
        calls: std::sync::atomic::AtomicUsize::new(0),
        gate: gate.clone(),
    }));
    let thread = h.agent(None, "agent question", &ws);
    h.events_until(&thread, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::ApprovalRequested { .. }))
    });
    py.run(
        &root,
        "fill-past-cap",
        &["--count", "59", "--prefix", "during"],
    );
    assert!(listed_ids(&store).contains(&thread));
    h.runtime
        .block_on(
            h.chat
                .decide(&thread, "c1", Decision::Reject { note: None }),
        )
        .unwrap();
    // While the model goes on (held), the 60th: the oldest listed thread
    // goes.
    py.run(
        &root,
        "fill-past-cap",
        &["--count", "1", "--prefix", "after"],
    );
    gate.notify_one();
    let events = h.turns_end(&thread, 1);
    let saved = events.iter().any(|event| {
        matches!(
            &event.kind,
            ConversationEventKind::TurnSaved { saved: true, .. }
        )
    });
    println!("I15 agent: saved {saved}");
    assert!(saved, "the answer was lost");
    assert!(
        listed_ids(&store).contains(&thread),
        "the thread is still listed"
    );
    let turns: Vec<String> = python_load(&py, &root, &thread, false)
        .iter()
        .map(|turn| text(turn, "text"))
        .collect();
    assert_eq!(turns, ["agent question", "the agent's answer"]);
    assert_archive_whole(&root, "I15 agent");
}

/// I14's sidecar half: Python supersedes a native agent turn, and the
/// agent chat's replay leaves out that turn's sidecar items (and its
/// words); a native regenerate is hidden by Python's `load_thread`.
#[test]
fn i14_agent_a_turn_python_supersedes_leaves_the_replay() {
    let Some(py) = python("I14-agent") else {
        return;
    };
    let dir = TempDir::new("agent-i14");
    let root = dir.path().join("store");
    let (h, store) = harness("agent-i14", &root);
    let ws = h.workspace();
    h.script(vec![
        call("read_file", json!({"path": "a.txt"}), "c_first"),
        say("native answer"),
    ]);
    let thread = h.agent(None, "native question", &ws);
    h.turns_end(&thread, 1);
    let question = store.load(&thread)[0].id.clone();
    let marked = py.run(
        &root,
        "supersede",
        &["--thread", &thread, "--from-turn", &question],
    );
    assert_eq!(int(&marked, "marked"), 2);
    let model = h.script(vec![say("second answer")]);
    h.agent(Some(&thread), "second question", &ws);
    h.turns_end(&thread, 2);
    let first = &model.calls()[0];
    let seen = format!("{:?}", first.input);
    assert!(!seen.contains("native question"), "{seen}");
    assert!(
        !first.input.iter().any(|item| matches!(
            item,
            InputItem::ToolResult { call_id, .. } if call_id == "c_first"
        )),
        "the superseded turn's tool items: {seen}"
    );
    assert_eq!(user_texts(first), ["second question"]);
    // A native regenerate, hidden by Python's load_thread.
    let regenerated = h.script(vec![say("regenerated")]);
    h.runtime
        .block_on(h.chat.regenerate(RegenerateRequest {
            conversation: thread.clone(),
            choice: "local".into(),
            shown: local(),
            mode: Mode::Agent,
        }))
        .unwrap();
    h.turns_end(&thread, 3);
    assert_eq!(user_texts(&regenerated.calls()[0]), ["second question"]);
    let visible: Vec<String> = python_load(&py, &root, &thread, false)
        .iter()
        .map(|turn| text(turn, "text"))
        .collect();
    println!("I14 agent: Python reads {visible:?}");
    assert_eq!(visible, ["second question", "regenerated"]);
}

// ------------------------------------------------- `created`, bit for bit

/// The `f64`s the `created` property runs over: the value row E11 met, the
/// edges, 10,000 times like Python's `time.time()` today (1.6e9 to 2.1e9,
/// every fraction bit random) and 10,000 finite doubles of any sign and
/// size, from a fixed seed.
fn created_values() -> Vec<f64> {
    let mut values = vec![
        f64::from_bits(0x41da_b04a_30ea_ee9f),
        0.0,
        -0.0,
        0.1 + 0.2,
        1e23,
        9_007_199_254_740_993.0,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::MAX,
        f64::MIN,
    ];
    // splitmix64: no crate, the same values every run.
    let mut state: u64 = 0x5eed_e11d_0000_0001;
    let mut next = move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let (low, high) = (1.6e9f64.to_bits(), 2.1e9f64.to_bits());
    for _ in 0..10_000 {
        values.push(f64::from_bits(low + next() % (high - low)));
    }
    while values.len() < 20_010 {
        let value = f64::from_bits(next());
        if value.is_finite() {
            values.push(value);
        }
    }
    values
}

/// Python's `json.dumps` of each value, one line each, from the repository's interpreter
/// (spec §16.1: no interpreter is a failure, never a skip). The child runs
/// isolated (`-I`), with an empty environment but `SystemRoot`, in a
/// temporary folder, under a deadline; one that overruns it is killed.
fn python_dumps(values: &[f64]) -> Vec<String> {
    use std::io::{Read, Write};
    let python = crate::chat::interop_tests::find_python().expect("the repository's Python");
    let dir = TempDir::new("created-python");
    let script = r"import json, struct, sys
for line in sys.stdin.read().split():
    sys.stdout.write(json.dumps(struct.unpack('>d', bytes.fromhex(line))[0]) + '\n')
";
    let mut command = std::process::Command::new(python);
    command
        .args(["-I", "-c", script])
        .env_clear()
        .current_dir(dir.path())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(root) = std::env::var_os("SystemRoot") {
        command.env("SystemRoot", root);
    }
    let mut child = command.spawn().expect("start Python");
    let input: String = values
        .iter()
        .map(|value| format!("{:016x}\n", value.to_bits()))
        .collect();
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(input.as_bytes()));
    let mut stdout = child.stdout.take().unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut text = String::new();
        let read = stdout.read_to_string(&mut text);
        let _ = sender.send(read.map(|_| text));
    });
    let text = match receiver.recv_timeout(Duration::from_secs(300)) {
        Ok(read) => read.expect("Python's output"),
        Err(_) => {
            let _ = child.kill();
            panic!("Python overran its deadline and was killed");
        }
    };
    let status = child.wait().unwrap();
    let mut errors = String::new();
    let _ = child.stderr.take().unwrap().read_to_string(&mut errors);
    assert!(status.success(), "Python failed: {errors}");
    writer.join().unwrap().expect("Python's input");
    text.lines().map(str::to_owned).collect()
}

/// Row E11d (§5.5, rows B5 and G4): `created` binds exactly. For 20,010
/// doubles (the value E11 met among them), the text Python's `json.dumps`
/// writes and the text native's `serde_json` writes each read back with
/// the same bits through every reader that meets a `created`: the
/// sidecar's (`Header`, `Meta`), a `serde_json::Value` (the reader's
/// archive record) and `pyjson` (the shared store). `same_created` takes
/// only the same bits: one unit apart is another conversation.
/// Mutants: `float_roundtrip` turned off; `same_created`'s tolerance of 2
/// units restored.
#[test]
fn created_reads_back_bit_for_bit_from_python_and_from_native() {
    use super::sidecar::{Header, Meta, same_created};
    use crate::chat::pyjson::{self, PyValue};
    // A copy of this crate outside the source repository has no repository
    // interpreter to require: say so and stop. Inside the repository a
    // missing interpreter stays a failure (spec §16.1).
    let helper = crate::chat::interop_tests::repo_root()
        .join("tools")
        .join("lattice_chat_interop.py");
    if std::env::var_os("LATTICE_PYTHON").is_none() && !helper.is_file() {
        println!(
            "created_reads_back_bit_for_bit_from_python_and_from_native: SKIPPED: not inside the source repository"
        );
        return;
    }
    let values = created_values();
    let python = python_dumps(&values);
    assert_eq!(python.len(), values.len(), "one line per value");
    let mut wrong = Vec::new();
    let mut same_text = 0;
    for (value, py_text) in values.iter().zip(&python) {
        let bits = value.to_bits();
        let native_text = serde_json::to_string(value).unwrap();
        if native_text == *py_text {
            same_text += 1;
        }
        for (writer, text) in [
            ("python", py_text.as_str()),
            ("native", native_text.as_str()),
        ] {
            let header: Header = serde_json::from_str(&format!(
                r#"{{"v":1,"id":"abc123def456","created":{text}}}"#
            ))
            .unwrap();
            let meta: Meta = serde_json::from_str(&format!(
                r#"{{"v":1,"id":"abc123def456","created":{text},"workspace":null,"mode":"ask","origin":"native"}}"#
            ))
            .unwrap();
            let value_read = serde_json::from_str::<Value>(text)
                .unwrap()
                .as_f64()
                .unwrap();
            let pyjson_read = match pyjson::loads(text) {
                Ok(PyValue::Float(read)) => read,
                other => panic!("{writer} {text}: pyjson read {other:?}"),
            };
            for (reader, read) in [
                ("Header", header.created),
                ("Meta", meta.created),
                ("Value", value_read),
                ("pyjson", pyjson_read),
            ] {
                if read.to_bits() != bits {
                    wrong.push(format!(
                        "{writer} {text} via {reader}: {bits:016x} read as {:016x}",
                        read.to_bits()
                    ));
                }
            }
            if !same_created(header.created, *value) {
                wrong.push(format!("{writer} {text}: not the same created"));
            }
        }
        if *value != f64::MAX && same_created(*value, value.next_up()) {
            wrong.push(format!("{bits:016x} and the next double bind as one"));
        }
    }
    println!(
        "created: {} values, {} wrong, the same text from both writers for {same_text}; first wrong: {:?}",
        values.len(),
        wrong.len(),
        &wrong[..wrong.len().min(5)]
    );
    assert!(
        wrong.is_empty(),
        "{} wrong: {:?}",
        wrong.len(),
        &wrong[..wrong.len().min(20)]
    );
}
