//! `lattice-chat-host`: the agent chat core with no window, for development
//! only (the chat core's spec §10.5 and §11.5; row F1).
//!
//! It is an example of `lattice-core` that needs the `dev-host` feature
//! (`required-features`), so `cargo build`, `cargo test` and every shipped
//! build leave it out; nothing packages it. `tools/lattice_egress_check.ps1`
//! builds and drives it:
//!
//! ```text
//! cargo run -p lattice-core --features dev-host --example lattice-chat-host -- \
//!     --root <a new, empty folder> --scenario <idle|turn|spin|connect> [--seconds N] [--port P]
//! ```
//!
//! **What it runs on.** Everything lives under `--root`, which must be absolute
//! and new or empty: the state root (`<root>/state`, so `<globals>` is
//! `<root>/state/globals`), a home (`<root>/home`, with an empty
//! `.gitconfig`), `ALELYON_HOME`, the temporary folder and a scratch git
//! repository (`<root>/work`). The core reads its environment through a
//! `MapEnv` holding only those names and Windows' own (`PATH`, `SystemRoot`,
//! `windir`, `PATHEXT`, `COMSPEC`): no key, no `FAM_ENV_PATH`, never the real
//! `~/.alelyon` or `globals/`. The transcript store is `MemoryTranscriptStore`
//! (the `dev-host` feature's), so the shared chat store is never opened. The
//! model is the agents crate's scripted model behind the model factory, so a
//! turn shown as Local reaches no server: the managed runtime is a stand-in that
//! hands out a loopback address the factory never calls. Every native dialog
//! (`ConfirmPort`) answers yes, as a scripted reader would.
//!
//! **Scenarios.**
//! - `idle` (E1's S0 and §11.5's reading): a folder attached and trusted, a
//!   conversation open and followed, and a `run_command` approval waiting.
//!   It then idles for `--seconds`, counting its runtime's wakes after a
//!   200 ms settle (CB1, live).
//! - `turn` (E1's S1): a Local agent turn that runs `read_file`, `grep` and an
//!   approved `git status` (Windows PowerShell 5.1, through the core's own
//!   approved-command path) in the scratch repository, then idles out
//!   `--seconds` in all.
//! - `spin` (§11.5's positive control): one thread named
//!   `lattice-spin-control` busy-loops for `--seconds`; the probe must read
//!   about one core and name this program as the thread's start module.
//! - `connect` (E1's positive control): one TCP connection to
//!   `127.0.0.1:--port`, a listener the egress script opened itself, held for
//!   `--seconds`. Only a loopback address is accepted.
//!
//! It prints one JSON object per line on standard output: `ready` (with its
//! process id), then `done` with what it saw.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use futures::future::BoxFuture;
use lattice_agents::testing::{ScriptedModel, ScriptedStep, assistant_message, function_call_json};
use lattice_agents::{ChatCompletionsConfig, model::Model};
use lattice_core::chat::memory::MemoryTranscriptStore;
use lattice_core::chat::store::uuid_ids;
use lattice_core::chat::transcript::TranscriptStore;
use lattice_core::chat::vocab::Target;
use lattice_core::clock::{Clock, system_clock};
use lattice_core::convo::agent::{AgentChat, AgentConfig, CapsFn};
use lattice_core::convo::caps::{ModelCaps, Tri};
use lattice_core::env::MapEnv;
use lattice_core::llama::files::{LlamaPaths, LocalModel};
use lattice_core::llama::{Lease, LlamaError, ManagedEndpoint, ManagedRuntime, Opened};
use lattice_core::models::ModelFactory;
use lattice_core::ports::{AttentionPort, ConfirmPort, ConfirmRequest};
use lattice_core::state::StateRoot;
use lattice_protocol::conversation::{
    Accepted, AgentChatService, ConversationEvent, ConversationEventKind, Decision, Mode,
    SendRequest,
};
use lattice_protocol::{Locality, Shown};
use serde_json::{Value, json};

/// CB1's settle: time in which the runtime did not wake.
const SETTLE: Duration = Duration::from_millis(200);
/// The longest any step is waited for.
const DEADLINE: Duration = Duration::from_secs(120);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scenario {
    Idle,
    Turn,
    Spin,
    Connect,
}

struct Args {
    root: PathBuf,
    scenario: Scenario,
    seconds: u64,
    port: Option<u16>,
}

fn parse_args() -> Result<Args, String> {
    let mut root = None;
    let mut scenario = None;
    let mut seconds = 60;
    let mut port = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--root" => root = Some(PathBuf::from(value)),
            "--scenario" => {
                scenario = Some(match value.as_str() {
                    "idle" => Scenario::Idle,
                    "turn" => Scenario::Turn,
                    "spin" => Scenario::Spin,
                    "connect" => Scenario::Connect,
                    other => return Err(format!("unknown scenario {other}")),
                })
            }
            "--seconds" => seconds = value.parse().map_err(|_| "--seconds: a number")?,
            "--port" => port = Some(value.parse().map_err(|_| "--port: a number")?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let root = root.ok_or("--root is required")?;
    if !root.is_absolute() {
        return Err("--root must be absolute".into());
    }
    if root.exists()
        && std::fs::read_dir(&root)
            .map_err(|e| format!("--root: {e}"))?
            .next()
            .is_some()
    {
        return Err("--root must be new or empty: the host never reuses a folder".into());
    }
    Ok(Args {
        root,
        scenario: scenario.ok_or("--scenario is required")?,
        seconds,
        port,
    })
}

fn emit(value: Value) {
    println!("{value}");
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(why) => {
            eprintln!("lattice-chat-host: {why}");
            return ExitCode::from(2);
        }
    };
    let result = match args.scenario {
        Scenario::Spin => spin(&args),
        Scenario::Connect => connect(&args),
        Scenario::Idle | Scenario::Turn => chat(&args),
    };
    match result {
        Ok(done) => {
            emit(done);
            ExitCode::SUCCESS
        }
        Err(why) => {
            emit(json!({"event": "failed", "why": why}));
            ExitCode::FAILURE
        }
    }
}

// ------------------------------------------------------- positive controls

/// §11.5's positive control: one named thread spinning for `seconds`.
fn spin(args: &Args) -> Result<Value, String> {
    std::fs::create_dir_all(&args.root).map_err(|e| e.to_string())?;
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let spins = Arc::new(AtomicU64::new(0));
    let count = spins.clone();
    let thread = std::thread::Builder::new()
        .name("lattice-spin-control".into())
        .spawn(move || {
            let mut n: u64 = 0;
            while !flag.load(Ordering::Relaxed) {
                n = std::hint::black_box(n.wrapping_add(1));
            }
            count.store(n, Ordering::SeqCst);
        })
        .map_err(|e| e.to_string())?;
    emit(json!({"event": "ready", "pid": std::process::id(), "scenario": "spin"}));
    std::thread::sleep(Duration::from_secs(args.seconds));
    stop.store(true, Ordering::Relaxed);
    thread.join().map_err(|_| "the spinning thread panicked")?;
    Ok(json!({"event": "done", "scenario": "spin", "iterations": spins.load(Ordering::SeqCst)}))
}

/// E1's positive control: a connection to a loopback listener the egress
/// script opened, held for `seconds`.
fn connect(args: &Args) -> Result<Value, String> {
    std::fs::create_dir_all(&args.root).map_err(|e| e.to_string())?;
    let port = args.port.ok_or("--port is required for connect")?;
    let address = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    emit(json!({"event": "ready", "pid": std::process::id(), "scenario": "connect"}));
    let stream = std::net::TcpStream::connect_timeout(&address, Duration::from_secs(10))
        .map_err(|e| format!("connect {address}: {e}"))?;
    let local = stream.local_addr().map_err(|e| e.to_string())?;
    std::thread::sleep(Duration::from_secs(args.seconds));
    drop(stream);
    Ok(
        json!({"event": "done", "scenario": "connect", "to": address.to_string(), "from": local.to_string()}),
    )
}

// ------------------------------------------------------------ the chat core

/// Every native dialog answered yes, each one recorded.
#[derive(Clone, Default)]
struct ScriptedReader {
    asked: Arc<Mutex<Vec<String>>>,
}

impl ConfirmPort for ScriptedReader {
    fn confirm(&self, request: ConfirmRequest) -> BoxFuture<'static, bool> {
        let name = format!("{request:?}");
        let name = name
            .split([' ', '{', '('])
            .next()
            .unwrap_or_default()
            .to_owned();
        self.asked.lock().unwrap().push(name);
        Box::pin(async { true })
    }
}

struct NoAttention;

impl AttentionPort for NoAttention {
    fn attention(&self, _conversation: &str) {}
}

/// The managed server's stand-in: an address nothing listens on, which the
/// model factory below never calls.
struct StandIn;

impl ManagedRuntime for StandIn {
    fn running(&self) -> Option<String> {
        None
    }

    fn failed(&self) -> Option<String> {
        None
    }

    fn open(&self, model: LocalModel) -> BoxFuture<'static, Result<Opened, LlamaError>> {
        Box::pin(async move {
            Ok(Opened {
                endpoint: ManagedEndpoint {
                    base_url: "http://127.0.0.1:9".into(),
                    alias: model.name,
                    token: "dev-host-token".into(),
                    binary_sha256: None,
                    generation: 1,
                },
                lease: Lease::detached(),
            })
        })
    }
}

fn call(name: &str, args: Value, id: &str) -> ScriptedStep {
    ScriptedStep::respond(vec![function_call_json(name, &args, id)])
}

fn say(text: &str) -> ScriptedStep {
    ScriptedStep::respond(vec![assistant_message(text)]).with_tokens(11, 7)
}

/// The scratch repository's git, for setup only: no global or system
/// configuration, a fixed identity.
fn setup_git(home: &Path, cwd: &Path, args: &[&str]) -> Result<(), String> {
    let output = Command::new("git")
        .args([
            "-c",
            "user.name=dev-host",
            "-c",
            "user.email=dev-host@localhost",
            "-c",
            "init.defaultBranch=main",
            "-c",
            "core.autocrlf=false",
        ])
        .args(args)
        .current_dir(cwd)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", home.join(".gitconfig"))
        .env("HOME", home)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .output()
        .map_err(|e| format!("setup git: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "setup git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn kind(event: &ConversationEvent) -> String {
    let text = format!("{:?}", event.kind);
    text.split([' ', '{', '('])
        .next()
        .unwrap_or_default()
        .to_owned()
}

fn chat(args: &Args) -> Result<Value, String> {
    let root = &args.root;
    let home = root.join("home");
    let tmp = root.join("tmp");
    let work = root.join("work");
    for dir in [&home, &tmp, &work] {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    write(&home.join(".gitconfig"), b"")?;

    // The environment the core sees: this folder's names and Windows' own.
    let mut env = MapEnv::new()
        .with("USERPROFILE", home.as_os_str())
        .with("HOME", home.as_os_str())
        .with("TEMP", tmp.as_os_str())
        .with("TMP", tmp.as_os_str())
        .with("ALELYON_HOME", root.join("alelyon_home").as_os_str());
    for name in ["PATH", "SystemRoot", "windir", "PATHEXT", "COMSPEC"] {
        if let Some(value) = std::env::var_os(name) {
            env.set(name, value);
        }
    }

    // The scratch repository.
    setup_git(&home, &work, &["init", "-q"])?;
    write(&work.join("a.txt"), b"alpha\nbeta\n")?;
    write(&work.join("notes.md"), b"# notes\nalpha again\n")?;
    write(&work.join(".gitignore"), b"*.log\n")?;
    setup_git(&home, &work, &["add", "a.txt", "notes.md", ".gitignore"])?;
    setup_git(&home, &work, &["commit", "-q", "-m", "one"])?;

    // The state root, a stand-in managed model, and the registry's one entry.
    let state = StateRoot::at(root.join("state"));
    std::fs::create_dir_all(&state.globals).map_err(|e| e.to_string())?;
    let paths = LlamaPaths::from_env(&env);
    // The stand-ins go under this run's root and nowhere else: never over a
    // real install in the reader's own `~/.alelyon`.
    for path in [&paths.binary, &paths.models_dir] {
        if !path.starts_with(root) {
            return Err(format!(
                "refused: {} is outside the scratch root {}",
                path.display(),
                root.display()
            ));
        }
    }
    write(&paths.binary, b"MZ")?;
    // The smallest GGUF header: the magic, version 3, no tensors, no keys.
    let mut gguf = b"GGUF".to_vec();
    gguf.extend_from_slice(&3u32.to_le_bytes());
    gguf.extend_from_slice(&0u64.to_le_bytes());
    gguf.extend_from_slice(&0u64.to_le_bytes());
    write(&paths.models_dir.join("dev-host-model.gguf"), &gguf)?;
    write(
        &state.globals.join("analyst_model.json"),
        b"{\"model\": \"dev-host-model\"}",
    )?;

    let unparks = Arc::new(AtomicU64::new(0));
    let counter = unparks.clone();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .on_thread_unpark(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        })
        .build()
        .map_err(|e| e.to_string())?;

    let steps = match args.scenario {
        Scenario::Idle => vec![
            call(
                "run_command",
                json!({"command": "git status", "timeout_s": 60}),
                "c_wait",
            ),
            say("unused"),
        ],
        _ => vec![
            call("read_file", json!({"path": "a.txt"}), "c_read"),
            call("grep", json!({"pattern": "alpha"}), "c_grep"),
            call(
                "run_command",
                json!({"command": "git status", "timeout_s": 60}),
                "c_git",
            ),
            say("Read a.txt, found alpha twice, and the tree is clean."),
        ],
    };
    let model: Arc<dyn Model> = Arc::new(ScriptedModel::new(steps));
    let asked = Arc::new(Mutex::new(Vec::<Value>::new()));
    let seen = asked.clone();
    let factory: ModelFactory = Arc::new(move |config: &ChatCompletionsConfig| {
        seen.lock().unwrap().push(json!({
            "base_url": config.base_url,
            "model": config.model,
            "local": config.local,
        }));
        Some(model.clone())
    });
    let caps: CapsFn = Arc::new(|resolution| ModelCaps {
        tools: match resolution.target {
            Target::Echo => Tri::No,
            _ => Tri::Yes,
        },
        vision: Tri::Unknown,
        context_tokens: None,
    });
    let clock: Clock = system_clock();
    let store: Arc<dyn TranscriptStore> =
        Arc::new(MemoryTranscriptStore::new(uuid_ids(), clock.clone()));
    let reader = ScriptedReader::default();
    let mut config = AgentConfig::new(
        state.clone(),
        Arc::new(env),
        Arc::new(StandIn),
        Arc::new(reader.clone()),
        Arc::new(NoAttention),
    );
    config.store = store;
    config.store_dir = "memory".into();
    config.development = true;
    config.clock = clock;
    config.model_factory = Some(factory);
    config.caps = Some(caps);
    let chat = AgentChat::new(config, runtime.handle().clone());

    let started = Instant::now();
    let view = runtime
        .block_on(chat.attach_native(None, work.clone()))
        .map_err(|r| format!("attach: {r:?}"))?;
    let workspace = view.workspace.id.clone();
    runtime
        .block_on(chat.trust(&workspace))
        .map_err(|r| format!("trust: {r:?}"))?;
    let accepted = runtime
        .block_on(chat.send(SendRequest {
            conversation: None,
            text: "Look at a.txt, find alpha, and run git status.".into(),
            choice: "local".into(),
            shown: Shown {
                locality: Locality::Local,
                label: "Local".into(),
            },
            mode: Mode::Agent,
            workspace: Some(workspace),
            edit_of: None,
            project: None,
            images: Vec::new(),
        }))
        .map_err(|r| format!("send: {r:?}"))?;
    let id = match accepted {
        Accepted::Started { conversation, .. } => conversation.id,
        other => return Err(format!("send: {other:?}")),
    };
    emit(
        json!({"event": "ready", "pid": std::process::id(), "scenario": format!("{:?}", args.scenario).to_lowercase()}),
    );

    // The interface's side: a follow held on the core's runtime for the
    // whole run, as a window showing the conversation would hold it.
    let events = Arc::new(Mutex::new(Vec::<ConversationEvent>::new()));
    let sink = events.clone();
    let mut stream = chat.follow(&id, 0).map_err(|r| format!("follow: {r:?}"))?;
    let follower = runtime.spawn(async move {
        while let Some(batch) = stream.next().await {
            sink.lock().unwrap().extend(batch);
        }
    });
    let wait_for = |what: &str, done: &dyn Fn(&[ConversationEvent]) -> bool| {
        let deadline = Instant::now() + DEADLINE;
        loop {
            if done(&events.lock().unwrap()) {
                return Ok(());
            }
            if Instant::now() > deadline {
                return Err(format!("timed out waiting for {what}"));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    };
    let approval = |call: &'static str| {
        move |all: &[ConversationEvent]| {
            all.iter().any(|event| {
                matches!(&event.kind, ConversationEventKind::ApprovalRequested { call_id, .. } if call_id.as_str() == call)
            })
        }
    };
    let ended = |all: &[ConversationEvent]| {
        all.iter()
            .any(|event| matches!(event.kind, ConversationEventKind::TurnEnded { .. }))
    };

    let mut idle_wakes = None;
    let mut idle_seconds = 0.0;
    match args.scenario {
        Scenario::Idle => {
            wait_for("the approval", &approval("c_wait"))?;
            // Settle: SETTLE with no wake of the runtime.
            let deadline = Instant::now() + DEADLINE;
            let mut last = unparks.load(Ordering::SeqCst);
            let mut quiet = Instant::now();
            while quiet.elapsed() < SETTLE {
                std::thread::sleep(Duration::from_millis(10));
                let now = unparks.load(Ordering::SeqCst);
                if now != last {
                    last = now;
                    quiet = Instant::now();
                }
                if Instant::now() > deadline {
                    return Err("the runtime never settled".into());
                }
            }
            emit(json!({"event": "idle", "pid": std::process::id()}));
            let window = Instant::now();
            let before = unparks.load(Ordering::SeqCst);
            std::thread::sleep(Duration::from_secs(args.seconds));
            idle_wakes = Some(unparks.load(Ordering::SeqCst) - before);
            idle_seconds = window.elapsed().as_secs_f64();
            let before_stop = unparks.load(Ordering::SeqCst);
            if !chat.stop(&id) {
                return Err("Stop found no running turn".into());
            }
            wait_for("the stopped turn's end", &ended)?;
            if unparks.load(Ordering::SeqCst) <= before_stop {
                return Err("positive control: Stop did not wake the runtime".into());
            }
        }
        _ => {
            wait_for("the git status approval", &approval("c_git"))?;
            runtime
                .block_on(chat.decide(&id, "c_git", Decision::Approve))
                .map_err(|r| format!("decide: {r:?}"))?;
            wait_for("the turn's end", &ended)?;
            let total = Duration::from_secs(args.seconds);
            if let Some(left) = total.checked_sub(started.elapsed()) {
                std::thread::sleep(left);
            }
        }
    }

    let texts: Vec<String> = chat_texts(&runtime, &chat, &id);
    runtime.block_on(chat.shutdown(Duration::from_secs(10)));
    follower.abort();
    let kinds: Vec<String> = events.lock().unwrap().iter().map(kind).collect();
    let asked_models = asked.lock().unwrap().clone();
    let dialogs = reader.asked.lock().unwrap().clone();
    Ok(json!({
        "event": "done",
        "scenario": format!("{:?}", args.scenario).to_lowercase(),
        "conversation": id,
        "events": kinds,
        "model_clients": asked_models,
        "dialogs": dialogs,
        "transcript": texts,
        "idle_window_seconds": idle_seconds,
        "idle_window_wakes": idle_wakes,
        "seconds": started.elapsed().as_secs_f64(),
    }))
}

fn chat_texts(runtime: &tokio::runtime::Runtime, chat: &AgentChat, id: &str) -> Vec<String> {
    match runtime.block_on(chat.open(id)) {
        Ok(snapshot) => snapshot
            .turns
            .iter()
            .map(|turn| format!("{:?}", turn))
            .map(|text| text.chars().take(160).collect())
            .collect(),
        Err(refusal) => vec![format!("open refused: {refusal:?}")],
    }
}
