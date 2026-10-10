//! `AgentChat` end to end (row E11): agent turns over a scratch repository,
//! the memory transcript store, a managed runtime that hands out a loopback
//! address nothing listens on, and a model factory that records every client
//! it is asked for and answers with a scripted model. Nothing reaches the
//! network, a GPU or the real `globals/`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use lattice_agents::ChatCompletionsConfig;
use lattice_agents::model::{InputItem, Model, ModelRequest};
use lattice_agents::testing::{ScriptedModel, ScriptedStep, assistant_message, function_call_json};
use lattice_protocol::conversation::{
    Accepted, AgentChatService, ConversationEvent, ConversationEventKind, Decision, DequeueOutcome,
    Mode, ReviewOp, SendRequest, TurnKind, TurnStatus, WorkspaceView,
};
use lattice_protocol::{Locality, Refusal, RefusalKind, Shown};
use serde_json::{Value, json};

use super::agent::{AgentChat, AgentConfig, words};
use super::caps::{ModelCaps, Tri};
use super::item::Item;
use super::sidecar::SidecarStore;
use crate::chat::core_tests::{Asked, FakeRuntime};
use crate::chat::memory::MemoryTranscriptStore;
use crate::chat::store::{IndexState, uuid_ids};
use crate::chat::transcript::TranscriptStore;
use crate::chat::vocab::Target;
use crate::clock::Clock;
use crate::env::MapEnv;
use crate::git::tests::Scratch;
use crate::llama::files::LlamaPaths;
use crate::models::ModelFactory;
use crate::ports::AttentionPort;
use crate::ports::fake::RecordingConfirm;
use crate::state::StateRoot;

/// A string the detector takes for a key, assembled so no source holds one.
pub(crate) fn secret() -> String {
    format!("sk-{}", "q7".repeat(12))
}

pub(crate) const HOSTED: &str = r#"{"version": 1, "endpoints": [
    {"id": "hosted", "label": "Hosted", "base_url": "https://hosted.example.test/v1",
     "model": "big", "api_key_name": "HOSTED_API_KEY", "enabled": true}
]}"#;

#[derive(Default)]
pub(crate) struct Attention(pub Mutex<Vec<String>>);

impl AttentionPort for Attention {
    fn attention(&self, conversation: &str) {
        self.0.lock().unwrap().push(conversation.to_owned());
    }
}

pub(crate) fn local() -> Shown {
    Shown {
        locality: Locality::Local,
        label: "Local".into(),
    }
}

pub(crate) fn hosted() -> Shown {
    Shown {
        locality: Locality::Remote,
        label: "Hosted".into(),
    }
}

/// One agent chat over one scratch repository.
pub(crate) struct H {
    pub scratch: Scratch,
    pub folder: PathBuf,
    pub state: StateRoot,
    pub env: MapEnv,
    pub runtime: tokio::runtime::Runtime,
    /// How often the runtime's workers woke (CB1, row E12).
    pub unparks: Arc<AtomicU64>,
    pub store: Arc<dyn TranscriptStore>,
    pub confirm: RecordingConfirm,
    pub attention: Arc<Attention>,
    pub asked: Arc<Mutex<Vec<Asked>>>,
    pub model: Arc<Mutex<Option<Arc<dyn Model>>>>,
    pub local: Arc<FakeRuntime>,
    pub chat: AgentChat,
}

pub(crate) fn tools_everywhere() -> super::agent::CapsFn {
    Arc::new(|resolution| ModelCaps {
        tools: match resolution.target {
            Target::Echo => Tri::No,
            _ => Tri::Yes,
        },
        vision: Tri::Unknown,
        context_tokens: None,
    })
}

impl H {
    pub(crate) fn new(tag: &str) -> Self {
        Self::with(tag, &[], |_| {})
    }

    pub(crate) fn with(
        tag: &str,
        extra: &[(&str, &str)],
        change: impl FnOnce(&mut AgentConfig),
    ) -> Self {
        let scratch = Scratch::new(tag);
        let folder = scratch.repo("work");
        let mut env = scratch.env().with("HOSTED_API_KEY", "fixture-hosted");
        for (name, value) in extra {
            env.set(name, *value);
        }
        let state = StateRoot::at(scratch.path().join("state"));
        std::fs::create_dir_all(&state.globals).unwrap();
        std::fs::write(state.globals.join("model_endpoints.json"), HOSTED).unwrap();
        let paths = LlamaPaths::from_env(&env);
        std::fs::create_dir_all(&paths.llama_dir).unwrap();
        std::fs::write(&paths.binary, b"MZ").unwrap();
        crate::llama::files::tests::gguf(&paths.models_dir.join("qwen3-8b.gguf"));
        std::fs::write(
            state.globals.join("analyst_model.json"),
            "{\"model\": \"qwen3-8b\"}",
        )
        .unwrap();
        let unparks = Arc::new(AtomicU64::new(0));
        let counter = unparks.clone();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .on_thread_unpark(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            })
            .build()
            .unwrap();
        let clock: Clock = crate::clock::system_clock();
        let store: Arc<dyn TranscriptStore> =
            Arc::new(MemoryTranscriptStore::new(uuid_ids(), clock.clone()));
        let local = Arc::new(FakeRuntime::default());
        let asked = Arc::new(Mutex::new(Vec::new()));
        let model: Arc<Mutex<Option<Arc<dyn Model>>>> = Arc::new(Mutex::new(None));
        let (seen, given) = (asked.clone(), model.clone());
        let factory: ModelFactory = Arc::new(move |config: &ChatCompletionsConfig| {
            seen.lock().unwrap().push(Asked {
                base_url: config.base_url.clone(),
                model: config.model.clone(),
                local: config.local,
            });
            given.lock().unwrap().clone()
        });
        let confirm = RecordingConfirm::answering(true);
        let attention = Arc::new(Attention::default());
        let env_arc: Arc<dyn crate::env::Env> = Arc::new(env.clone());
        let mut config = AgentConfig::new(
            state.clone(),
            env_arc,
            local.clone(),
            Arc::new(confirm.clone()),
            attention.clone(),
        );
        config.store = store.clone();
        config.store_dir = "memory".into();
        config.development = true;
        config.clock = clock;
        config.model_factory = Some(factory);
        config.caps = Some(tools_everywhere());
        config.runner = Some(Arc::new(scratch.runner()));
        change(&mut config);
        let store = config.store.clone();
        let chat = AgentChat::new(config, runtime.handle().clone());
        Self {
            scratch,
            folder,
            state,
            env,
            runtime,
            unparks,
            store,
            confirm,
            attention,
            asked,
            model,
            local,
            chat,
        }
    }

    pub(crate) fn answer_with(&self, model: Arc<dyn Model>) {
        *self.model.lock().unwrap() = Some(model);
    }

    /// A scripted model answering these steps, kept for its calls.
    pub(crate) fn script(&self, steps: Vec<ScriptedStep>) -> Arc<ScriptedModel> {
        let model = Arc::new(ScriptedModel::new(steps));
        self.answer_with(model.clone());
        model
    }

    pub(crate) fn attach(&self) -> WorkspaceView {
        self.runtime
            .block_on(self.chat.attach_native(None, self.folder.clone()))
            .unwrap()
    }

    /// Attach and trust the scratch repository.
    pub(crate) fn workspace(&self) -> String {
        let view = self.attach();
        let id = view.workspace.id.clone();
        self.runtime.block_on(self.chat.trust(&id)).unwrap();
        id
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn send(
        &self,
        conversation: Option<&str>,
        text: &str,
        choice: &str,
        shown: Shown,
        mode: Mode,
        workspace: Option<&str>,
    ) -> Result<Accepted, Refusal> {
        self.runtime.block_on(self.chat.send(SendRequest {
            conversation: conversation.map(str::to_owned),
            text: text.into(),
            choice: choice.into(),
            shown,
            mode,
            workspace: workspace.map(str::to_owned),
            edit_of: None,
            project: None,
            images: Vec::new(),
        }))
    }

    /// An agent send on Local in Agent mode; the conversation's id.
    pub(crate) fn agent(&self, conversation: Option<&str>, text: &str, workspace: &str) -> String {
        match self
            .send(
                conversation,
                text,
                "local",
                local(),
                Mode::Agent,
                Some(workspace),
            )
            .unwrap()
        {
            Accepted::Started { conversation, .. } => conversation.id,
            other => panic!("{other:?}"),
        }
    }

    /// Every event from the start until `done` holds (300 s at most: a bound
    /// on a hang, not a measure of speed, as a loaded machine has taken over
    /// 60 s to start a real PowerShell or browser).
    pub(crate) fn events_until(
        &self,
        id: &str,
        done: impl Fn(&[ConversationEvent]) -> bool,
    ) -> Vec<ConversationEvent> {
        let mut stream = self.chat.follow(id, 0).unwrap();
        self.runtime.block_on(async {
            let mut all = Vec::new();
            let deadline = tokio::time::Instant::now() + Duration::from_secs(300);
            while !done(&all) {
                match tokio::time::timeout_at(deadline, stream.next()).await {
                    Ok(Some(batch)) => all.extend(batch),
                    _ => panic!("timed out waiting; events so far: {all:#?}"),
                }
            }
            all
        })
    }

    /// Every event until `n` turns have ended.
    pub(crate) fn turns_end(&self, id: &str, n: usize) -> Vec<ConversationEvent> {
        self.events_until(id, |events| ended(events) >= n)
    }

    pub(crate) fn sidecar_items(&self, id: &str) -> Vec<Item> {
        SidecarStore::new(self.state.native_chat_dir())
            .read_items(id)
            .map(|log| log.items)
            .unwrap_or_default()
    }

    pub(crate) fn asked(&self) -> Vec<Asked> {
        self.asked.lock().unwrap().clone()
    }

    pub(crate) fn texts(&self, id: &str) -> Vec<String> {
        self.store
            .load(id)
            .into_iter()
            .map(|turn| turn.text)
            .collect()
    }

    pub(crate) fn snapshot(&self) -> String {
        let state = self.store.list();
        let mut text = format!("{state:?}");
        if let IndexState::Rows(rows) = &state {
            for row in rows {
                text.push_str(&format!("\n{}: {:?}", row.id, self.store.load(&row.id)));
            }
        }
        text
    }
}

pub(crate) fn ended(events: &[ConversationEvent]) -> usize {
    events
        .iter()
        .filter(|event| matches!(event.kind, ConversationEventKind::TurnEnded { .. }))
        .count()
}

pub(crate) fn statuses(events: &[ConversationEvent]) -> Vec<TurnStatus> {
    events
        .iter()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::TurnEnded { status, .. } => Some(*status),
            _ => None,
        })
        .collect()
}

pub(crate) fn user_texts(request: &ModelRequest) -> Vec<String> {
    request
        .input
        .iter()
        .filter_map(|item| match item {
            InputItem::User(text) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

pub(crate) fn call(name: &str, args: Value, id: &str) -> ScriptedStep {
    ScriptedStep::respond(vec![function_call_json(name, &args, id)])
}

pub(crate) fn say(text: &str) -> ScriptedStep {
    ScriptedStep::respond(vec![assistant_message(text)]).with_tokens(11, 7)
}

/// Every file under `root`, with its bytes.
pub(crate) fn files(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let bytes = std::fs::read(&path).unwrap_or_else(|_| b"(locked)".to_vec());
                out.insert(path.to_string_lossy().into_owned(), bytes);
            }
        }
    }
    out
}

// ------------------------------------------------------------- the turn

/// §2.4 end to end, TF9 and AF8: an agent turn reads a file and answers; the
/// rules lead as `input[0]` and never in the system prompt; the user's text
/// reaches the model exactly once; the tool trail is in the sidecar, the
/// shared record holds only the question and the answer (S18, D9), and the
/// tokens are the sums (D10).
/// Mutants: rules placed in the system prompt (TF9); the replay including
/// the current user record (AF8, E11b's m3).
#[test]
fn an_agent_turn_reads_answers_and_records_its_trail() {
    let h = H::new("agent-turn");
    std::fs::write(h.folder.join("AGENTS.md"), "RULES-MARKER: be brief.\n").unwrap();
    let ws = h.workspace();
    let model = h.script(vec![
        call("read_file", json!({"path": "a.txt"}), "call_1").with_tokens(5, 2),
        say("a.txt holds the letter a."),
    ]);
    let id = h.agent(None, "What is in a.txt?", &ws);
    let events = h.turns_end(&id, 1);
    assert_eq!(statuses(&events), [TurnStatus::Completed]);
    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    // TF9.
    match &calls[0].input[0] {
        InputItem::User(text) => assert!(text.contains("RULES-MARKER"), "{text}"),
        other => panic!("{other:?}"),
    }
    assert!(!calls[0].system.contains("RULES-MARKER"));
    assert!(calls[0].system.contains("PowerShell"), "the agent prompt");
    // AF8: the question once.
    for request in &calls {
        let n = user_texts(request)
            .iter()
            .filter(|text| text.contains("What is in a.txt?"))
            .count();
        assert_eq!(n, 1, "{:?}", request.input);
    }
    let tools: Vec<String> = calls[0]
        .tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect();
    assert_eq!(tools.len(), 23, "Agent mode offers every tool: {tools:?}");
    assert!(
        matches!(&calls[1].input.last(), Some(InputItem::ToolResult { output, .. }) if output.contains("1\ta"))
    );
    // The shared record: the question and the answer only.
    let stored = h.store.load(&id);
    assert_eq!(stored.len(), 2);
    assert_eq!(stored[1].text, "a.txt holds the letter a.");
    assert!(stored[1].tools.is_empty() && stored[1].facts.is_empty());
    assert!(stored[1].unsupported.is_empty(), "D9");
    assert_eq!(stored[1].provider, "llamacpp:qwen3-8b");
    assert_eq!(
        (stored[1].prompt_tokens, stored[1].completion_tokens),
        (Some(16), Some(9))
    );
    // The sidecar: the trail.
    let kinds: Vec<&'static str> = h
        .sidecar_items(&id)
        .iter()
        .map(|item| match item {
            Item::TurnStart { .. } => "turn_start",
            Item::ToolCall { .. } => "tool_call",
            Item::ToolResult { .. } => "tool_result",
            Item::TurnEnd { .. } => "turn_end",
            _ => "other",
        })
        .filter(|kind| *kind != "other")
        .collect();
    assert_eq!(
        kinds,
        ["turn_start", "tool_call", "tool_result", "turn_end"]
    );
    // One client for the turn, local.
    assert_eq!(h.asked().len(), 1);
    assert!(h.asked()[0].local);
    // The trace, in the run store's format, under <native>/chat/runs.
    let runs = files(&h.state.chat_runs_dir());
    assert!(
        runs.keys().any(|name| name.ends_with(".run.json")),
        "{runs:?}"
    );
    assert!(runs.keys().any(|name| name.ends_with(".events.jsonl")));
}

/// The mode filter (§7.1): Ask mode offers the read tools and
/// `ask_question` only.
#[test]
fn ask_mode_offers_no_staging_tool_and_no_command() {
    let h = H::new("agent-ask-mode");
    let ws = h.workspace();
    let model = h.script(vec![say("ok")]);
    let id = match h
        .send(None, "look", "local", local(), Mode::Ask, Some(&ws))
        .unwrap()
    {
        Accepted::Started {
            conversation,
            turn_kind,
            ..
        } => {
            assert_eq!(turn_kind, TurnKind::Agent);
            conversation.id
        }
        other => panic!("{other:?}"),
    };
    h.turns_end(&id, 1);
    let names: Vec<String> = model.calls()[0]
        .tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect();
    assert_eq!(
        names,
        [
            "list_dir",
            "glob",
            "read_file",
            "grep",
            "ask_question",
            "git_status",
            "remember",
            "forget",
            "spawn_agent",
            "suggest_task",
            "withdraw_task",
            "write_artifact",
            "read_artifact",
            "update_todos",
            "propose_plan"
        ]
    );
}

/// FT3: Agent mode in an untrusted folder is refused before anything is
/// written; Ask mode reads.
#[test]
fn agent_mode_needs_a_trusted_folder() {
    let h = H::new("agent-untrusted");
    let ws = h.attach().workspace.id;
    h.script(vec![say("ok")]);
    let refusal = h
        .send(None, "edit", "local", local(), Mode::Agent, Some(&ws))
        .unwrap_err();
    assert_eq!(refusal.message, words::TRUST_FIRST);
    assert_eq!(h.snapshot(), "Absent");
    assert!(h.asked().is_empty());
}

/// §2.4 step 3: no workspace is the plain turn, its events forwarded.
#[test]
fn without_a_workspace_the_turn_is_plain() {
    let h = H::new("agent-plain");
    h.script(vec![say("plain answer")]);
    let accepted = h
        .send(None, "hello", "local", local(), Mode::Agent, None)
        .unwrap();
    let id = match accepted {
        Accepted::Started {
            conversation,
            turn_kind,
            ..
        } => {
            assert_eq!(turn_kind, TurnKind::Plain);
            conversation.id
        }
        other => panic!("{other:?}"),
    };
    let events = h.turns_end(&id, 1);
    assert!(events.iter().any(|event| matches!(
        &event.kind,
        ConversationEventKind::TurnSaved { turn, saved: true } if turn.text == "plain answer"
    )));
    assert_eq!(h.texts(&id), ["hello", "plain answer"]);
}

/// §4.5: an endpoint's tools are Unknown and behave as Yes; a first
/// tool-bearing request answered 400 ends the turn with the sentence, the
/// pair is cached as No, and the next send is a plain turn. A managed model
/// with no probe record is probed at its first Agent-mode turn (LR8′): prose
/// fails the probe, the turn says so, and the next send is plain, told so.
/// Mutant: no cache (the next send is an agent turn again).
#[test]
fn a_model_that_refuses_tools_falls_back_to_plain_turns() {
    let h = H::with("agent-caps", &[], |config| config.caps = None);
    let ws = h.workspace();
    h.script(vec![ScriptedStep::error(
        lattice_agents::ModelError::Status(400),
    )]);
    let id = match h
        .send(
            None,
            "hi",
            "endpoint:hosted",
            hosted(),
            Mode::Agent,
            Some(&ws),
        )
        .unwrap()
    {
        Accepted::Started { conversation, .. } => conversation.id,
        other => panic!("{other:?}"),
    };
    let events = h.turns_end(&id, 1);
    assert!(events.iter().any(|event| matches!(
        &event.kind,
        ConversationEventKind::Error { message } if message == words::NO_TOOLS
    )));
    h.script(vec![say("plain")]);
    match h
        .send(
            Some(&id),
            "again",
            "endpoint:hosted",
            hosted(),
            Mode::Agent,
            None,
        )
        .unwrap()
    {
        Accepted::Started { turn_kind, .. } => assert_eq!(turn_kind, TurnKind::Plain),
        other => panic!("{other:?}"),
    }
    h.turns_end(&id, 2);
    // A managed model with no probe record: probed, failed by prose, said so.
    h.script(vec![say("I would call it.")]);
    let other = match h
        .send(None, "q", "local", local(), Mode::Agent, Some(&ws))
        .unwrap()
    {
        Accepted::Started {
            conversation,
            turn_kind,
            ..
        } => {
            assert_eq!(turn_kind, TurnKind::Agent);
            conversation.id
        }
        other => panic!("{other:?}"),
    };
    let events = h.turns_end(&other, 1);
    assert!(events.iter().any(|event| matches!(
        &event.kind,
        ConversationEventKind::Error { message } if message == words::LOCAL_NO_TOOLS
    )));
    // Then plain, and said so.
    h.script(vec![say("plain local")]);
    match h
        .send(Some(&other), "q2", "local", local(), Mode::Agent, None)
        .unwrap()
    {
        Accepted::Started { turn_kind, .. } => assert_eq!(turn_kind, TurnKind::Plain),
        other => panic!("{other:?}"),
    }
    let events = h.turns_end(&other, 2);
    assert!(events.iter().any(|event| matches!(
        &event.kind,
        ConversationEventKind::Notice { text } if text == super::caps::PLAIN_FALLBACK
    )));
}

// --------------------------------------------- queue, steer, stop, continue

/// §7.7 and §4.4: Stop during a question ends the turn Stopped, saves "" with
/// the sentence, and the question no longer waits; a send while the turn
/// runs is queued and goes after it.
#[test]
fn stop_ends_a_waiting_turn_and_a_queued_message_goes_next() {
    let h = H::new("agent-stop");
    let ws = h.workspace();
    h.script(vec![
        call("ask_question", json!({"question": "Which file?"}), "q1"),
        say("second answer"),
    ]);
    let id = h.agent(None, "start", &ws);
    h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::Question { .. }))
    });
    assert_eq!(*h.attention.0.lock().unwrap(), std::slice::from_ref(&id));
    let queued = h
        .send(Some(&id), "next", "local", local(), Mode::Agent, Some(&ws))
        .unwrap();
    assert!(
        matches!(queued, Accepted::Queued { position: 1, .. }),
        "{queued:?}"
    );
    assert!(h.chat.stop(&id));
    let events = h.turns_end(&id, 2);
    assert_eq!(
        statuses(&events),
        [TurnStatus::Stopped, TurnStatus::Completed]
    );
    let stored = h.store.load(&id);
    let texts: Vec<&str> = stored.iter().map(|turn| turn.text.as_str()).collect();
    assert_eq!(texts, ["start", "", "next", "second answer"]);
    assert!(stored[1].cancelled);
    assert_eq!(stored[1].error, crate::chat::answer::NOTHING_WRITTEN);
    assert!(
        h.chat.answer(&id, "q1", "late".into()).is_err(),
        "the question no longer waits"
    );
    assert!(events.iter().any(|event| matches!(
        &event.kind,
        ConversationEventKind::Dequeued {
            outcome: DequeueOutcome::Sent { .. },
            ..
        }
    )));
}

/// §7.7 through the service: the answer reaches the model as "The user
/// answered: ...", recorded as an `Answer` item.
#[test]
fn a_question_is_answered_through_the_service() {
    let h = H::new("agent-question");
    let ws = h.workspace();
    let model = h.script(vec![
        call(
            "ask_question",
            json!({"question": "Proceed?", "options": ["yes", "no"]}),
            "q1",
        ),
        say("done"),
    ]);
    let id = h.agent(None, "go", &ws);
    h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::Question { .. }))
    });
    h.chat.answer(&id, "q1", "yes".into()).unwrap();
    h.turns_end(&id, 1);
    let calls = model.calls();
    assert!(
        matches!(calls[1].input.last(), Some(InputItem::ToolResult { output, .. }) if output == "The user answered: yes")
    );
    assert!(
        h.sidecar_items(&id)
            .iter()
            .any(|item| matches!(item, Item::Answer { text, .. } if text == "yes"))
    );
}

/// §3.3, AF7 and A4a end to end: a steer during a turn is taken before the
/// next model call; a steer sent while the final model call answers is not
/// lost: it is queued and answered in a later turn.
/// Mutant: unsent steers dropped at the end of the turn.
#[test]
fn steers_reach_the_next_model_call_or_the_queue() {
    let h = H::new("agent-steer");
    let ws = h.workspace();
    let model = Arc::new(
        ScriptedModel::new(vec![
            call("ask_question", json!({"question": "Wait"}), "q1"),
            say("first"),
            say("answer to the late steer"),
        ])
        .with_delay(Duration::from_millis(1_500)),
    );
    h.answer_with(model.clone());
    let id = h.agent(None, "start", &ws);
    h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::Question { .. }))
    });
    h.runtime
        .block_on(h.chat.steer(&id, "EARLY-STEER".into()))
        .unwrap();
    h.chat.answer(&id, "q1", "ok".into()).unwrap();
    // Steer into the final call while it answers (it waits 1.5 s first).
    let started = std::time::Instant::now();
    while model.calls().len() < 2 {
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the final call never started"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    h.runtime
        .block_on(h.chat.steer(&id, "LATE-STEER".into()))
        .unwrap();
    let events = h.turns_end(&id, 2);
    let calls = model.calls();
    assert!(
        user_texts(&calls[1])
            .iter()
            .any(|text| text == "EARLY-STEER")
    );
    println!("{:?}", calls.iter().map(user_texts).collect::<Vec<_>>());
    assert!(
        calls[2..]
            .iter()
            .any(|request| user_texts(request).iter().any(|text| text == "LATE-STEER")),
        "the late steer was answered in a later turn"
    );
    assert!(events.iter().any(|event| matches!(&event.kind, ConversationEventKind::Steered { text } if text == "EARLY-STEER")));
    assert!(h.texts(&id).contains(&"LATE-STEER".to_owned()));
}

/// §4.4 and AF8's Continue half: after MaxTurns, Continue replays up to the
/// last tool result with no new user item and no new user turn.
#[test]
fn continue_after_max_turns_adds_no_user_item() {
    let h = H::with("agent-continue", &[], |config| config.max_turns = 1);
    let ws = h.workspace();
    let model = h.script(vec![
        call("read_file", json!({"path": "a.txt"}), "call_1"),
        say("finished after continuing"),
    ]);
    let id = h.agent(None, "read it", &ws);
    let events = h.turns_end(&id, 1);
    assert_eq!(statuses(&events), [TurnStatus::MaxTurns]);
    let accepted = h
        .runtime
        .block_on(h.chat.continue_turn(&id, self::local()))
        .unwrap();
    assert!(matches!(accepted, Accepted::Started { .. }));
    let events = h.turns_end(&id, 2);
    assert_eq!(statuses(&events)[1], TurnStatus::Completed);
    let calls = model.calls();
    assert_eq!(user_texts(&calls[1]), ["read it"], "no new user item");
    assert!(matches!(
        calls[1].input.last(),
        Some(InputItem::ToolResult { .. })
    ));
    let texts = h.texts(&id);
    assert_eq!(
        texts.iter().filter(|text| *text == "read it").count(),
        1,
        "no new user turn: {texts:?}"
    );
    assert_eq!(texts.last().unwrap(), "finished after continuing");
}

/// T2: `pin`, `set_mode` and `archive` are refused while a turn runs.
#[test]
fn t2_nothing_moves_the_target_while_a_turn_runs() {
    let h = H::new("agent-t2");
    let ws = h.workspace();
    h.script(vec![
        call("ask_question", json!({"question": "Wait"}), "q1"),
        say("done"),
    ]);
    let id = h.agent(None, "start", &ws);
    h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::Question { .. }))
    });
    for refusal in [
        h.runtime.block_on(h.chat.pin(&id, "auto")).unwrap_err(),
        h.runtime
            .block_on(h.chat.set_mode(&id, Mode::Ask))
            .unwrap_err(),
        h.runtime.block_on(h.chat.archive(&id)).unwrap_err(),
    ] {
        assert_eq!(refusal.kind, RefusalKind::Conflict);
    }
    h.chat.answer(&id, "q1", "go".into()).unwrap();
    h.turns_end(&id, 1);
    assert_eq!(h.asked().len(), 1, "TF7: one client for the turn");
}

/// E3's "the agent hears it at its next model call": an Undo's note reaches
/// the next turn in its leading user item.
#[test]
fn a_review_note_reaches_the_agent_at_its_next_turn() {
    let h = H::new("agent-review-note");
    let ws = h.workspace();
    let model = h.script(vec![
        call(
            "edit_file",
            json!({"path": "a.txt", "old_string": "a", "new_string": "b"}),
            "e1",
        ),
        say("staged"),
        say("understood"),
    ]);
    let id = h.agent(None, "change a.txt", &ws);
    let events = h.turns_end(&id, 1);
    let change = events
        .iter()
        .find_map(|event| match &event.kind {
            ConversationEventKind::Staged { change, .. } => Some(change.clone()),
            _ => None,
        })
        .expect("a Staged event");
    let outcome = h
        .runtime
        .block_on(h.chat.review(
            &id,
            vec![ReviewOp::Undo {
                change,
                hunks: None,
                note: Some("NOTE-MARKER keep it as it was".into()),
            }],
        ))
        .unwrap();
    assert_eq!(outcome.results.len(), 1);
    h.agent(Some(&id), "and now?", &ws);
    h.turns_end(&id, 2);
    let calls = model.calls();
    let lead = user_texts(&calls[2]).join("\n");
    assert!(lead.contains("NOTE-MARKER"), "{lead}");
    assert!(lead.contains("The user undid `a.txt`."), "{lead}");
    assert_eq!(
        std::fs::read(h.folder.join("a.txt")).unwrap(),
        b"a\n",
        "nothing written"
    );
}

/// §4.6 and AF9: Lattice closing while a call waits on its approval leaves
/// the call in the sidecar; reopened, the turn is `Interrupted`, and
/// Continue's replay holds the call and the synthesised "closed before
/// deciding" result exactly once.
/// Mutant: tool items written only by `add_items` (E11b's m4 for the reply).
#[test]
fn af9_a_turn_killed_while_waiting_continues_with_the_call_answered_once() {
    let h = H::new("agent-af9");
    let ws = h.workspace();
    h.script(vec![call(
        "run_command",
        json!({"command": "Write-Output hi"}),
        "c1",
    )]);
    let id = h.agent(None, "run it", &ws);
    h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::ApprovalRequested { .. }))
    });
    // Lattice closes: the runtime goes, and the service with it.
    let H {
        scratch,
        folder,
        state,
        env,
        runtime,
        unparks,
        store,
        asked,
        model,
        local,
        chat,
        confirm,
        attention,
    } = h;
    runtime.shutdown_background();
    drop(chat);
    let items: Vec<Item> = SidecarStore::new(state.native_chat_dir())
        .read_items(&id)
        .unwrap()
        .items;
    assert!(
        items
            .iter()
            .any(|item| matches!(item, Item::ToolCall { call_id, .. } if call_id == "c1"))
    );
    assert!(
        !items
            .iter()
            .any(|item| matches!(item, Item::TurnEnd { .. }))
    );
    // A new process over the same store and state.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let mut config = AgentConfig::new(
        state.clone(),
        Arc::new(env.clone()),
        local.clone(),
        Arc::new(confirm.clone()),
        attention.clone(),
    );
    config.store = store.clone();
    config.development = true;
    config.caps = Some(tools_everywhere());
    config.runner = Some(Arc::new(scratch.runner()));
    let (seen, given) = (asked.clone(), model.clone());
    config.model_factory = Some(Arc::new(move |config: &ChatCompletionsConfig| {
        seen.lock().unwrap().push(Asked {
            base_url: config.base_url.clone(),
            model: config.model.clone(),
            local: config.local,
        });
        given.lock().unwrap().clone()
    }));
    let chat = AgentChat::new(config, runtime.handle().clone());
    let h = H {
        scratch,
        folder,
        state,
        env,
        runtime,
        unparks,
        store,
        confirm,
        attention,
        asked,
        model,
        local,
        chat,
    };
    let snapshot = h.runtime.block_on(h.chat.open(&id)).unwrap();
    assert!(snapshot.events.iter().any(|event| matches!(
        event.kind,
        ConversationEventKind::TurnEnded {
            status: TurnStatus::Interrupted,
            ..
        }
    )));
    let model = h.script(vec![say("continued")]);
    h.runtime
        .block_on(h.chat.continue_turn(&id, self::local()))
        .unwrap();
    h.events_until(&id, |events| {
        events.iter().any(|event| {
            matches!(
                event.kind,
                ConversationEventKind::TurnEnded {
                    status: TurnStatus::Completed,
                    ..
                }
            )
        })
    });
    let first = &model.calls()[0];
    let calls: Vec<&InputItem> = first
        .input
        .iter()
        .filter(|item| matches!(item, InputItem::Assistant { tool_calls, .. } if tool_calls.iter().any(|call| call.call_id == "c1")))
        .collect();
    let results: Vec<&InputItem> = first
        .input
        .iter()
        .filter(|item| matches!(item, InputItem::ToolResult { call_id, output } if call_id == "c1" && output == super::replay::CLOSED_BEFORE))
        .collect();
    assert_eq!((calls.len(), results.len()), (1, 1), "{:#?}", first.input);
    assert!(user_texts(first) == ["run it"], "no new user item");
}

/// §5.5 through the agent chat's send (G4's owed caller): a record left by
/// an earlier conversation with the same id is moved aside, never used.
#[test]
fn a_send_binds_the_sidecar_by_created_and_moves_an_earlier_record_aside() {
    let ids: crate::chat::store::IdSource = Arc::new(|| "abc123def456".to_owned());
    let h = H::with("agent-binding", &[], |config| {
        config.store = Arc::new(MemoryTranscriptStore::new(
            ids,
            crate::clock::system_clock(),
        ));
    });
    let sidecars = SidecarStore::new(h.state.native_chat_dir());
    drop(
        sidecars
            .open_for_writing(
                "abc123def456",
                1.25,
                super::sidecar::NewMeta {
                    workspace: None,
                    mode: Mode::Ask,
                    origin: lattice_protocol::conversation::Origin::Native,
                },
            )
            .unwrap(),
    );
    let ws = h.workspace();
    h.script(vec![say("ok")]);
    let id = h.agent(None, "hello", &ws);
    assert_eq!(id, "abc123def456");
    h.turns_end(&id, 1);
    let moved = sidecars.root().join("conversations").join(format!(
        "abc123def456~{}",
        super::sidecar::created_tag(1.25)
    ));
    assert!(moved.is_dir(), "the earlier record was moved aside");
    let created = match h.store.list() {
        IndexState::Rows(rows) => rows[0].created,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        sidecars.read_items(&id).unwrap().header.created.to_bits(),
        created.to_bits()
    );
}

/// S21 through the service (G4's owed caller): Archive is the reader's
/// archive, listed with its reason; there is no delete.
#[test]
fn archive_is_the_readers_archive_listed_with_its_reason() {
    let h = H::new("agent-archive");
    let ws = h.workspace();
    h.script(vec![say("ok")]);
    let id = h.agent(None, "hello", &ws);
    h.turns_end(&id, 1);
    h.runtime.block_on(h.chat.archive(&id)).unwrap();
    let list = h.runtime.block_on(h.chat.list()).unwrap();
    assert!(list.conversations.is_empty());
    assert_eq!(list.archived.len(), 1);
    assert_eq!(
        list.archived[0].reason,
        lattice_protocol::conversation::ArchiveReason::Reader
    );
    let key = list.archived[0].key.clone();
    let summary = h.runtime.block_on(h.chat.unarchive(key)).unwrap();
    assert_eq!(summary.id, id);
}

/// E7/E8 through the service: an approved command's events, its effect in
/// the folder, and its whole output recorded with the turn and read by
/// window; Approve opens the native dialog; a dialog refused is a reject.
#[test]
fn an_approved_command_runs_with_its_events_effect_and_output() {
    let h = H::new("agent-command");
    let ws = h.workspace();
    let model = h.script(vec![
        call(
            "run_command",
            // A bound, not a speed: PowerShell has taken 35 s to start on a
            // loaded machine.
            json!({"command": "Set-Content -Path made.txt -Value made; Write-Output COMMAND-OUTPUT", "timeout_s": 300}),
            "c1",
        ),
        say("ran it"),
    ]);
    let id = h.agent(None, "run something", &ws);
    h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::ApprovalRequested { .. }))
    });
    assert!(h.chat.list_needs_you(&h.runtime, &id));
    h.runtime
        .block_on(h.chat.decide(&id, "c1", Decision::Approve))
        .unwrap();
    let events = h.turns_end(&id, 1);
    assert!(
        h.confirm
            .asked()
            .iter()
            .any(|request| matches!(request, crate::ports::ConfirmRequest::RunCommand { .. }))
    );
    let kinds: Vec<&ConversationEventKind> = events.iter().map(|event| &event.kind).collect();
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ConversationEventKind::CommandStarted { .. }))
    );
    assert!(kinds.iter().any(|kind| matches!(
        kind,
        ConversationEventKind::CommandExited { code: Some(0), .. }
    )));
    assert!(kinds.iter().any(|kind| matches!(
        kind,
        ConversationEventKind::CommandEffect { files, .. } if files.iter().any(|file| file.path == "made.txt")
    )), "{kinds:#?}");
    assert!(
        matches!(model.calls()[1].input.last(), Some(InputItem::ToolResult { output, .. }) if output.contains("COMMAND-OUTPUT"))
    );
    let lines = h
        .runtime
        .block_on(h.chat.read_lines(
            lattice_protocol::conversation::ViewRef::Output {
                conversation: id.clone(),
                call_id: "c1".into(),
            },
            1,
            10,
        ))
        .unwrap();
    assert!(
        lines
            .lines
            .iter()
            .any(|line| line.contains("COMMAND-OUTPUT")),
        "{lines:?}"
    );
    let changes = h.runtime.block_on(h.chat.changes(&id)).unwrap();
    assert!(
        changes
            .changes
            .iter()
            .any(|change| change.path == "made.txt")
    );
}

/// §9.3 and CF15 through the service: Approve with the native dialog
/// refusing is a reject with no note; nothing runs.
#[test]
fn approve_refused_in_the_dialog_is_a_reject_and_nothing_runs() {
    let h = H::new("agent-command-refused");
    let ws = h.workspace();
    let model = h.script(vec![
        call(
            "run_command",
            json!({"command": "Set-Content -Path never.txt -Value x"}),
            "c1",
        ),
        say("not run"),
    ]);
    let id = h.agent(None, "run", &ws);
    h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::ApprovalRequested { .. }))
    });
    *h.confirm.answer.lock().unwrap() = false;
    h.runtime
        .block_on(h.chat.decide(&id, "c1", Decision::Approve))
        .unwrap();
    h.turns_end(&id, 1);
    assert!(!h.folder.join("never.txt").exists());
    assert!(
        matches!(model.calls()[1].input.last(), Some(InputItem::ToolResult { output, .. }) if output == "Tool execution was not approved.")
    );
    assert!(
        h.runtime
            .block_on(h.chat.decide(&id, "c1", Decision::Approve))
            .is_err(),
        "no longer pending"
    );
}

/// A3a end to end: when a call that needs approval and one that does not
/// share a response, the free call's result is recorded first, as the SDK
/// orders them.
#[test]
fn a3a_the_free_calls_result_is_recorded_before_the_approved_ones() {
    let h = H::new("agent-a3a");
    let ws = h.workspace();
    h.script(vec![
        ScriptedStep::respond(vec![
            function_call_json(
                "run_command",
                &json!({"command": "Write-Output x"}),
                "c_cmd",
            ),
            function_call_json("read_file", &json!({"path": "a.txt"}), "c_read"),
        ]),
        say("both done"),
    ]);
    let id = h.agent(None, "both", &ws);
    h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::ApprovalRequested { .. }))
    });
    h.runtime
        .block_on(h.chat.decide(
            &id,
            "c_cmd",
            Decision::Reject {
                note: Some("no".into()),
            },
        ))
        .unwrap();
    h.turns_end(&id, 1);
    let results: Vec<String> = h
        .sidecar_items(&id)
        .into_iter()
        .filter_map(|item| match item {
            Item::ToolResult { call_id, .. } => Some(call_id),
            _ => None,
        })
        .collect();
    assert_eq!(results, ["c_read", "c_cmd"]);
}

impl AgentChat {
    /// The list's `needs_you` for `id` (tests).
    fn list_needs_you(&self, runtime: &tokio::runtime::Runtime, id: &str) -> bool {
        runtime
            .block_on(self.list())
            .unwrap()
            .conversations
            .iter()
            .any(|summary| summary.id == id && summary.needs_you)
    }
}

/// §5.5: a conversation's own record is bound again after its `created`
/// went through `meta.json` and back. The value is the one row E11's
/// binding test met by chance, which `serde_json`'s default parser read one
/// unit off; with `float_roundtrip` (row E11d) it reads back exactly, and
/// the binding is exact: one unit apart is another conversation.
/// Mutants: E11c's m13 (exact comparison without the feature); E11d's
/// feature off, and the tolerance restored.
#[test]
fn a_record_binds_again_after_its_created_round_trips_through_json() {
    let created = f64::from_bits(0x41da_b04a_30ea_ee9f);
    let back: f64 = serde_json::from_str(&serde_json::to_string(&created).unwrap()).unwrap();
    println!(
        "created {:x}, read back {:x}",
        created.to_bits(),
        back.to_bits()
    );
    assert_eq!(back.to_bits(), created.to_bits(), "read back exactly");
    let dir = crate::testkit::TempDir::new("created-round-trip");
    let store = SidecarStore::new(dir.path());
    let new = || super::sidecar::NewMeta {
        workspace: None,
        mode: Mode::Ask,
        origin: lattice_protocol::conversation::Origin::Native,
    };
    drop(
        store
            .open_for_writing("abc123def456", created, new())
            .unwrap(),
    );
    let (_, binding) = store
        .open_for_writing("abc123def456", created, new())
        .unwrap();
    assert_eq!(binding, super::sidecar::Binding::Bound);
    assert!(super::sidecar::same_created(created, back));
    assert!(!super::sidecar::same_created(created, created.next_up()));
    assert!(!super::sidecar::same_created(created, created + 1e-3));
    // Exact: a thread one unit later is another conversation, and the
    // earlier record is moved aside under its own bits.
    let (_, binding) = store
        .open_for_writing("abc123def456", created.next_up(), new())
        .unwrap();
    assert_eq!(
        binding,
        super::sidecar::Binding::MovedAside("abc123def456~41dab04a30eaee9f".into())
    );
}

/// A native dialog port that answers yes at once, except RunCommand, which
/// says it was asked and then waits for the test's word.
struct HeldRunCommand {
    asked: Mutex<Option<std::sync::mpsc::Sender<()>>>,
    go: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}

impl crate::ports::ConfirmPort for HeldRunCommand {
    fn confirm(
        &self,
        request: crate::ports::ConfirmRequest,
    ) -> futures::future::BoxFuture<'static, bool> {
        if !matches!(request, crate::ports::ConfirmRequest::RunCommand { .. }) {
            return Box::pin(async { true });
        }
        let asked = self.asked.lock().unwrap().take();
        let go = self.go.lock().unwrap().take();
        Box::pin(async move {
            if let Some(asked) = asked {
                let _ = asked.send(());
            }
            if let Some(go) = go {
                let _ = go.await;
            }
            true
        })
    }
}

/// The first decision on a pending command wins (the verifier's PLAUSIBLE
/// "approval double record"): a page Reject that arrives while an Approve
/// waits in the RunCommand dialog decides the call; the Approve, once its
/// dialog says yes, finds the call no longer pending, and records, runs
/// and makes ready nothing. Exactly one ApprovalDecided (the Reject) and one
/// ApprovalResolved reach the record and the events.
/// Falsifier: fails on the code before this fix (a second ApprovalDecided{Approve}).
#[test]
fn a_reject_while_approve_waits_in_the_dialog_is_the_only_decision() {
    let (asked_tx, asked_rx) = std::sync::mpsc::channel();
    let (go_tx, go_rx) = tokio::sync::oneshot::channel();
    let port = Arc::new(HeldRunCommand {
        asked: Mutex::new(Some(asked_tx)),
        go: Mutex::new(Some(go_rx)),
    });
    let h = H::with("agent-first-decision", &[], move |config| {
        config.confirm = port;
    });
    let ws = h.workspace();
    let model = h.script(vec![
        call(
            "run_command",
            json!({"command": "Set-Content -Path never.txt -Value x"}),
            "c1",
        ),
        say("not run"),
    ]);
    let id = h.agent(None, "run", &ws);
    h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::ApprovalRequested { .. }))
    });
    let approve = h.runtime.spawn(h.chat.decide(&id, "c1", Decision::Approve));
    asked_rx
        .recv_timeout(Duration::from_secs(120))
        .expect("the RunCommand dialog opened");
    h.runtime
        .block_on(h.chat.decide(
            &id,
            "c1",
            Decision::Reject {
                note: Some("no".into()),
            },
        ))
        .unwrap();
    let _ = go_tx.send(());
    let late = h.runtime.block_on(approve).unwrap();
    assert!(late.is_err(), "the later Approve is refused: {late:?}");
    let events = h.turns_end(&id, 1);
    assert!(!h.folder.join("never.txt").exists(), "nothing ran");
    assert!(
        matches!(model.calls()[1].input.last(), Some(InputItem::ToolResult { output, .. }) if output.contains("not approved") || output.contains("no")),
        "{:?}",
        model.calls()[1].input.last()
    );
    let decided: Vec<_> = h
        .sidecar_items(&id)
        .into_iter()
        .filter_map(|item| match item {
            Item::ApprovalDecided { decision, .. } => Some(decision),
            _ => None,
        })
        .collect();
    assert_eq!(
        decided,
        vec![Decision::Reject {
            note: Some("no".into())
        }],
        "one decision, the first"
    );
    let resolved = events
        .iter()
        .filter(|event| matches!(event.kind, ConversationEventKind::ApprovalResolved { .. }))
        .count();
    assert_eq!(resolved, 1);
}
