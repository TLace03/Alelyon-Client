//! The managed llama.cpp server, against a stub (the chat core's spec
//! §22.2: LF1–LF5; CB1 for its idle timer; LR8's `/props`).
//!
//! The server is `lattice-llama-stub` (`tests/stub/llama_stub.rs`), started
//! through the runtime exactly as the real `llama-server` would be. Nothing
//! here starts the real server, uses a GPU, or reads or writes the real
//! `~/.alelyon`: the environment is a `MapEnv` whose home, temporary folder
//! and state root are this test's own folders, and whose
//! `ALELYON_LLAMA_SERVER` names the stub. Every request goes to 127.0.0.1.
//! Only processes these tests start are ended, through the runtime's Job.

#![cfg(windows)]

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use futures::future::BoxFuture;
use lattice_agents::model::{InputItem, ModelEvent, ModelRequest, ModelSettings};
use lattice_core::chat::vocab::{self, LocalView, Target};
use lattice_core::convo::caps::{Tri, managed_caps};
use lattice_core::env::MapEnv;
use lattice_core::keys::KeyStore;
use lattice_core::llama::files::{BinaryProblem, LlamaPaths, LocalModel, list_models};
use lattice_core::llama::server::{LlamaRuntime, RuntimeConfig, command};
use lattice_core::llama::settings::Settings;
use lattice_core::llama::{LlamaError, ManagedEndpoint, Opened, probes};
use lattice_core::models::{self, ModelFactory};
use lattice_core::net::{
    HttpGet, HttpRequest, HttpResponse, LoopbackHttp, NetError, is_loopback_url,
};
use lattice_core::state::StateRoot;
use serde_json::{Value, json};

const STUB: &str = env!("CARGO_BIN_EXE_lattice-llama-stub");

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "lattice-core-llama-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Only this test's own temporary folder.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Every request the runtime makes, seen before it is made.
struct Recording {
    inner: LoopbackHttp,
    urls: Mutex<Vec<String>>,
}

impl HttpGet for Recording {
    fn get(&self, request: HttpRequest) -> BoxFuture<'static, Result<HttpResponse, NetError>> {
        self.urls.lock().unwrap().push(request.url.clone());
        self.inner.get(request)
    }
}

struct Setup {
    root: Scratch,
    env: Arc<MapEnv>,
    state: StateRoot,
    http: Arc<Recording>,
}

impl Setup {
    fn new(tag: &str) -> Self {
        Self::with_binary(tag, Path::new(STUB))
    }

    fn with_binary(tag: &str, binary: &Path) -> Self {
        let root = Scratch::new(tag);
        let home = root.0.join("home");
        let temp = root.0.join("tmp");
        std::fs::create_dir_all(&temp).unwrap();
        let mut env = MapEnv::new()
            .with("USERPROFILE", home.as_os_str())
            .with("TEMP", temp.as_os_str())
            .with(lattice_core::llama::files::BINARY_ENV, binary.as_os_str())
            // What must never reach the server.
            .with("LLAMA_ARG_CTX_SIZE", "7")
            .with("OLLAMA_HOST", "0.0.0.0")
            .with("OLLAMA_MODELS", r"D:\ollama")
            .with("ANTHROPIC_API_KEY", "lattice-lf3-sentinel-key")
            .with("HOSTED_API_KEY", "fixture-hosted");
        // What a Windows program needs to start at all (X7 passes them on).
        for name in ["SystemRoot", "PATH", "windir"] {
            if let Some(value) = std::env::var_os(name) {
                env.set(name, value);
            }
        }
        let state = StateRoot::at(root.0.join("root"));
        std::fs::create_dir_all(&state.globals).unwrap();
        let http = Arc::new(Recording {
            inner: LoopbackHttp::new().unwrap(),
            urls: Mutex::new(Vec::new()),
        });
        Self {
            root,
            env: Arc::new(env),
            state,
            http,
        }
    }

    fn paths(&self) -> LlamaPaths {
        LlamaPaths::from_env(self.env.as_ref())
    }

    fn settings(&self, json: &str) {
        let paths = self.paths();
        std::fs::create_dir_all(&paths.llama_dir).unwrap();
        std::fs::write(paths.settings_file(), json).unwrap();
    }

    /// A GGUF model in the models folder, with the stub's behaviour for it.
    fn model(&self, name: &str, behaviour: Value) -> LocalModel {
        let dir = self.paths().models_dir;
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.gguf"));
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        std::fs::write(
            dir.join(format!("{name}.gguf.stub.json")),
            behaviour.to_string(),
        )
        .unwrap();
        list_models(&dir)
            .into_iter()
            .find(|model| model.name == name)
            .unwrap()
    }

    fn choose(&self, name: &str) {
        std::fs::write(
            self.state.globals.join("analyst_model.json"),
            format!("{{\"model\": \"{name}\"}}"),
        )
        .unwrap();
    }

    fn runtime(&self, handle: tokio::runtime::Handle) -> LlamaRuntime {
        let mut config = RuntimeConfig::new(self.env.clone(), &self.state).unwrap();
        config.http = self.http.clone();
        config.start_timeout = Duration::from_secs(20);
        config.poll_gap = Duration::from_millis(50);
        LlamaRuntime::new(config, handle)
    }

    /// A runtime with the test's settings, then `edit` applied.
    fn runtime_with(
        &self,
        handle: tokio::runtime::Handle,
        edit: impl FnOnce(&mut RuntimeConfig),
    ) -> LlamaRuntime {
        let mut config = RuntimeConfig::new(self.env.clone(), &self.state).unwrap();
        config.http = self.http.clone();
        config.start_timeout = Duration::from_secs(20);
        config.poll_gap = Duration::from_millis(50);
        edit(&mut config);
        LlamaRuntime::new(config, handle)
    }

    /// A runtime whose launches take `ports` in turn, then free ones.
    fn runtime_on_ports(&self, handle: tokio::runtime::Handle, ports: Vec<u16>) -> LlamaRuntime {
        let mut config = RuntimeConfig::new(self.env.clone(), &self.state).unwrap();
        config.http = self.http.clone();
        config.start_timeout = Duration::from_secs(20);
        config.poll_gap = Duration::from_millis(50);
        let queue = Mutex::new(std::collections::VecDeque::from(ports));
        config.ports = Arc::new(move || match queue.lock().unwrap().pop_front() {
            Some(port) => Ok(port),
            None => lattice_core::net::free_loopback_port(),
        });
        LlamaRuntime::new(config, handle)
    }

    fn urls(&self) -> Vec<String> {
        self.http.urls.lock().unwrap().clone()
    }
}

fn tokio() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

/// What the stub recorded when it started: its argv, environment and bound
/// address, one per launch.
fn launches(model: &LocalModel) -> Vec<Value> {
    let dir = model.path.parent().unwrap();
    let prefix = format!("{}.gguf.stub-", model.name);
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with(&prefix) && name.ends_with(".json") && !name.ends_with("stub.json") {
            let text = std::fs::read_to_string(entry.path()).unwrap();
            found.push(serde_json::from_str(&text).unwrap());
        }
    }
    found
}

fn argv_of(launch: &Value) -> Vec<String> {
    launch["argv"]
        .as_array()
        .unwrap()
        .iter()
        .map(|arg| arg.as_str().unwrap().to_owned())
        .collect()
}

fn key_file(launch: &Value) -> PathBuf {
    let argv = argv_of(launch);
    let at = argv.iter().position(|arg| arg == "--api-key-file").unwrap();
    PathBuf::from(&argv[at + 1])
}

/// Wait at most `limit` for `done`.
fn eventually(limit: Duration, mut done: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < limit {
        if done() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    done()
}

/// A raw HTTP/1.1 request to the stub, on loopback.
fn raw(port: u16, method: &str, path: &str, token: Option<&str>, body: &str) -> u16 {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    let auth = token
        .map(|t| format!("Authorization: Bearer {t}\r\n"))
        .unwrap_or_default();
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut answer = String::new();
    let _ = stream.read_to_string(&mut answer);
    answer
        .split(' ')
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0)
}

fn port_of(endpoint: &ManagedEndpoint) -> u16 {
    endpoint
        .base_url
        .rsplit(':')
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

/// LF2: a request without the token is refused, and the server listens on
/// loopback only (its command line and its bound address say 127.0.0.1).
#[test]
fn a_request_without_the_token_is_refused_and_the_server_is_on_loopback() {
    let setup = Setup::new("lf2");
    let model = setup.model("tiny", json!({}));
    let rt = tokio();
    let runtime = setup.runtime(rt.handle().clone());
    let opened: Opened = rt.block_on(runtime.open(model.clone())).unwrap();
    let port = port_of(&opened.endpoint);
    let token = opened.endpoint.token.expose().to_owned();
    let chat =
        r#"{"model": "tiny", "stream": false, "messages": [{"role": "user", "content": "hi"}]}"#;
    let outcomes = [
        (
            "GET /props without a token",
            raw(port, "GET", "/props", None, ""),
        ),
        (
            "GET /props with a wrong token",
            raw(port, "GET", "/props", Some("wrong"), ""),
        ),
        (
            "POST chat without a token",
            raw(port, "POST", "/v1/chat/completions", None, chat),
        ),
        (
            "GET /v1/models without a token",
            raw(port, "GET", "/v1/models", None, ""),
        ),
        (
            "GET /props with the token",
            raw(port, "GET", "/props", Some(&token), ""),
        ),
        (
            "POST chat with the token",
            raw(port, "POST", "/v1/chat/completions", Some(&token), chat),
        ),
        (
            "GET /health without a token",
            raw(port, "GET", "/health", None, ""),
        ),
    ];
    println!("{outcomes:?}");
    assert_eq!(
        outcomes.iter().map(|(_, code)| *code).collect::<Vec<_>>(),
        [401, 401, 401, 401, 200, 200, 200]
    );
    let launch = &launches(&model)[0];
    let argv = argv_of(launch);
    let host = argv.iter().position(|arg| arg == "--host").unwrap();
    assert_eq!(argv[host + 1], "127.0.0.1");
    assert_eq!(launch["bound"], format!("127.0.0.1:{port}"));
    assert!(
        setup.urls().iter().all(|url| is_loopback_url(url)),
        "{:?}",
        setup.urls()
    );
    drop(opened);
    rt.block_on(runtime.shutdown());
}

/// LF3: the launch is exactly LR6's: the command line is Python's
/// `ManagedServer.command` (`llama/llama_server.json`) plus
/// `--jinja`; no `LLAMA_ARG_*` or `OLLAMA_*` name and no key reaches the
/// environment; the token is not on the command line; the tools flag is
/// absent; and a binary in an Ollama install is never started. LR6′'s three
/// cases (the Vulkan device-selection names) follow it as tests of their own.
#[test]
fn the_launch_is_exactly_lr6s() {
    let setup = Setup::new("lf3");
    setup.settings(r#"{"ctx_size": 4096, "gpu_layers": 12, "parallel": 2}"#);
    let model = setup.model("tiny", json!({}));
    let rt = tokio();
    let runtime = setup.runtime(rt.handle().clone());
    let opened = rt.block_on(runtime.open(model.clone())).unwrap();
    let launch = &launches(&model)[0];
    let argv = argv_of(launch);
    let expected: Vec<String> = command(
        Path::new(STUB),
        port_of(&opened.endpoint),
        &model,
        &Settings {
            ctx_size: 4096,
            gpu_layers: 12,
            parallel: 2,
            ..Settings::default()
        },
        &key_file(launch),
    )
    .into_iter()
    .map(|arg: OsString| arg.to_string_lossy().into_owned())
    .collect();
    assert_eq!(argv, expected);

    // Python's command line, recorded from `ManagedServer.command`, with the
    // placeholders put back: the native one is it plus `--jinja`.
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/parity/llama/llama_server.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let case = &golden["command"];
    let python: Vec<String> = case["argv"]
        .as_array()
        .unwrap()
        .iter()
        .map(|arg| arg.as_str().unwrap().to_owned())
        .collect();
    let input = &case["input"];
    let native: Vec<String> = command(
        Path::new(input["binary"].as_str().unwrap()),
        input["port"].as_u64().unwrap() as u16,
        &LocalModel {
            name: input["name"].as_str().unwrap().into(),
            path: input["model"].as_str().unwrap().into(),
            size: 1,
        },
        &Settings {
            ctx_size: input["ctx_size"].as_u64().unwrap(),
            gpu_layers: input["gpu_layers"].as_u64().unwrap(),
            parallel: input["parallel"].as_u64().unwrap(),
            ..Settings::default()
        },
        Path::new(input["key_file"].as_str().unwrap()),
    )
    .into_iter()
    .map(|arg| arg.to_string_lossy().into_owned())
    .collect();
    assert_eq!(native[..native.len() - 1], python[..]);
    assert_eq!(native.last().map(String::as_str), Some("--jinja"));

    let tools = format!("--{}", "tools");
    assert!(!argv.iter().any(|arg| arg.starts_with(&tools)), "{argv:?}");
    let token = opened.endpoint.token.expose().to_owned();
    assert!(
        !argv.iter().any(|arg| arg.contains(&token)),
        "the token is not on the command line"
    );
    let env = launch["env"].as_object().unwrap();
    let names: Vec<&String> = env.keys().collect();
    println!("the server's environment: {names:?}");
    for name in &names {
        let upper = name.to_uppercase();
        assert!(
            !upper.starts_with("LLAMA_ARG_") && !upper.starts_with("OLLAMA_"),
            "{name}"
        );
        assert!(!upper.contains("API_KEY"), "{name}");
    }
    assert!(
        !launch["env"]
            .to_string()
            .contains("lattice-lf3-sentinel-key")
    );
    assert!(!launch["env"].to_string().contains(&token));
    assert!(
        names
            .iter()
            .any(|name| name.as_str() == "NoDefaultCurrentDirectoryInExePath")
    );
    drop(opened);
    rt.block_on(runtime.shutdown());

    // A copy of the stub in a folder named for Ollama is never started.
    let ollama_dir = setup.root.0.join("Programs").join("Ollama");
    std::fs::create_dir_all(&ollama_dir).unwrap();
    let ollama_binary = ollama_dir.join("llama-server.exe");
    std::fs::copy(STUB, &ollama_binary).unwrap();
    let refused = Setup::with_binary("lf3-ollama", &ollama_binary);
    let model = refused.model("tiny", json!({}));
    let rt = tokio();
    let runtime = refused.runtime(rt.handle().clone());
    let outcome = rt.block_on(runtime.open(model.clone()));
    assert_eq!(
        outcome.err(),
        Some(LlamaError::Binary(BinaryProblem::Ollama))
    );
    assert!(launches(&model).is_empty(), "nothing was started");
    assert!(refused.urls().is_empty(), "nothing was asked");
}

/// LF3's cases for LR6′ (amendment 1b): the Vulkan-looking names (`VK_*`,
/// `GGML_VK_*`) the stub's environment held when it started, with their
/// values, for a server the runtime started with `extra` set in the core's
/// environment. Each case expects at least one name, so a server
/// environment without the pass-through fails all three.
fn vulkan_names_the_server_got(tag: &str, extra: &[(&str, &str)]) -> Vec<(String, String)> {
    let mut setup = Setup::new(tag);
    let mut env = (*setup.env).clone();
    for (name, value) in extra {
        env.set(name, *value);
    }
    setup.env = Arc::new(env);
    let model = setup.model("tiny", json!({}));
    let rt = tokio();
    let runtime = setup.runtime(rt.handle().clone());
    let opened = rt.block_on(runtime.open(model.clone())).unwrap();
    let launch = &launches(&model)[0];
    let mut got: Vec<(String, String)> = launch["env"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(name, _)| {
            let upper = name.to_uppercase();
            upper.starts_with("VK_") || upper.starts_with("GGML_VK_")
        })
        .map(|(name, value)| (name.clone(), value.as_str().unwrap().to_owned()))
        .collect();
    got.sort();
    drop(opened);
    rt.block_on(runtime.shutdown());
    got
}

/// The six device-selection names, each with a value of its own.
const VULKAN_DEVICE_SELECTION: [(&str, &str); 6] = [
    ("GGML_VK_VISIBLE_DEVICES", "0"),
    ("VK_DRIVER_FILES", r"C:\drivers\amd_icd64.json"),
    ("VK_ICD_FILENAMES", r"C:\drivers\amd_icd64-old.json"),
    ("VK_LOADER_DEVICE_ID_FILTER", "0x7550"),
    ("VK_LOADER_DRIVERS_DISABLE", "*intel*"),
    ("VK_LOADER_DRIVERS_SELECT", "*amd*"),
];

fn sorted(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = pairs
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect();
    out.sort();
    out
}

/// LF3, LR6′ case 1: each device-selection name that is set in the core's
/// environment reaches the server, with its value.
/// Mutant: no pass-through.
#[test]
fn lf3_the_vulkan_device_names_reach_the_server_when_set() {
    let got = vulkan_names_the_server_got("lf3-vk-set", &VULKAN_DEVICE_SELECTION);
    assert_eq!(got, sorted(&VULKAN_DEVICE_SELECTION));
}

/// LF3, LR6′ case 2: no other `VK_*` name reaches the server, even when it
/// is set beside the six.
/// Mutants: no pass-through; every `VK_*` name passed.
#[test]
fn lf3_no_other_vulkan_name_reaches_the_server() {
    let mut extra = VULKAN_DEVICE_SELECTION.to_vec();
    extra.extend([
        ("VK_INSTANCE_LAYERS", "VK_LAYER_KHRONOS_validation"),
        ("VK_LAYER_PATH", r"C:\layers"),
        ("VK_ADD_DRIVER_FILES", r"C:\drivers\extra.json"),
        ("VK_LOADER_DEBUG", "all"),
        ("VK_LOADER_LAYERS_ENABLE", "*"),
        ("GGML_VK_DISABLE_F16", "1"),
    ]);
    let got = vulkan_names_the_server_got("lf3-vk-other", &extra);
    assert_eq!(got, sorted(&VULKAN_DEVICE_SELECTION));
}

/// LF3, LR6′ case 3: a device-selection name that is not set in the core's
/// environment does not appear in the server's.
/// Mutants: no pass-through; a name passed (empty) when it is not set.
#[test]
fn lf3_a_vulkan_device_name_that_is_not_set_never_appears() {
    let only = [
        ("GGML_VK_VISIBLE_DEVICES", "1"),
        ("VK_LOADER_DEVICE_ID_FILTER", "0x7550"),
    ];
    let got = vulkan_names_the_server_got("lf3-vk-unset", &only);
    assert_eq!(got, sorted(&only));
}

fn key_gone(launch: &Value) -> bool {
    let key = key_file(launch);
    !key.exists() && !key.parent().unwrap().exists()
}

/// LF4: the token file, and its folder, are gone once the server is ready
/// (LR6a), and after a stop, an idle stop,
/// a crash, a failed start and a drop of the runtime.
#[test]
fn the_token_file_is_gone_after_every_kind_of_stop() {
    let rt = tokio();
    let mut seen = Vec::new();

    // A stop (shutdown).
    let setup = Setup::new("lf4-stop");
    let model = setup.model("tiny", json!({}));
    let runtime = setup.runtime(rt.handle().clone());
    let opened = rt.block_on(runtime.open(model.clone())).unwrap();
    let launch = launches(&model).remove(0);
    // LR6a: llama.cpp reads the key only at start-up, so the file and its
    // folder are gone as soon as the server is ready, while it still runs;
    // the token it read still opens its doors.
    // Mutant: the removal at readiness dropped.
    seen.push(("ready, still running", key_gone(&launch)));
    assert_eq!(runtime.running().as_deref(), Some("tiny"));
    let props = rt.block_on(runtime.props(&opened.endpoint));
    assert!(props.is_ok(), "the server read the token: {props:?}");
    drop(opened);
    rt.block_on(runtime.shutdown());
    seen.push(("stop", key_gone(&launch)));

    // An idle stop.
    let setup = Setup::new("lf4-idle");
    setup.settings(r#"{"idle_seconds": 0.2}"#);
    let model = setup.model("tiny", json!({}));
    let runtime = setup.runtime(rt.handle().clone());
    drop(rt.block_on(runtime.open(model.clone())).unwrap());
    let launch = launches(&model).remove(0);
    let stopped = eventually(Duration::from_secs(5), || runtime.running().is_none());
    seen.push((
        "idle stop",
        stopped && eventually(Duration::from_secs(5), || key_gone(&launch)),
    ));

    // A crash after it was ready.
    let setup = Setup::new("lf4-crash");
    let model = setup.model("tiny", json!({"exit_after_ms": 1500}));
    let runtime = setup.runtime(rt.handle().clone());
    let opened = rt.block_on(runtime.open(model.clone())).unwrap();
    let launch = launches(&model).remove(0);
    // The key went at readiness; the crash must still be noticed.
    let noticed = eventually(Duration::from_secs(8), || runtime.failed().is_some());
    seen.push(("crash", noticed && key_gone(&launch)));
    assert_eq!(
        runtime.failed().as_deref(),
        Some("llama.cpp's server stopped on its own.")
    );
    drop(opened);

    // A failed start: the server exits while it loads.
    let setup = Setup::new("lf4-load");
    let model = setup.model("tiny", json!({"exit_during_load": true, "load_ms": 300}));
    let runtime = setup.runtime(rt.handle().clone());
    let outcome = rt.block_on(runtime.open(model.clone()));
    assert_eq!(outcome.err(), Some(LlamaError::ExitedWhileLoading));
    let launch = launches(&model).remove(0);
    seen.push(("failed start", key_gone(&launch)));

    // A drop of the runtime, with the server running and a lease out.
    let setup = Setup::new("lf4-drop");
    let model = setup.model("tiny", json!({}));
    let runtime = setup.runtime(rt.handle().clone());
    let opened = rt.block_on(runtime.open(model.clone())).unwrap();
    let port = port_of(&opened.endpoint);
    let launch = launches(&model).remove(0);
    drop(runtime);
    drop(opened);
    seen.push(("drop", key_gone(&launch)));
    let gone = rt.block_on(LoopbackHttp::new().unwrap().get(HttpRequest {
        url: format!("http://127.0.0.1:{port}/health"),
        bearer: None,
        timeout: Duration::from_secs(2),
    }));
    seen.push(("drop ends the process", gone.is_err()));

    println!("{seen:?}");
    assert!(seen.iter().all(|(_, gone)| *gone), "{seen:?}");
}

/// Stream one chat completion through the real Chat Completions client.
fn chat(rt: &tokio::runtime::Runtime, endpoint: &ManagedEndpoint) -> String {
    let built = models::build_managed(endpoint, None).unwrap();
    rt.block_on(async {
        let mut stream = built.model.stream(ModelRequest {
            system: String::new(),
            input: vec![InputItem::User("hi".into())],
            tools: Vec::new(),
            settings: ModelSettings {
                temperature: Some(0.2),
                ..ModelSettings::default()
            },
        });
        let mut text = String::new();
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                ModelEvent::TextDelta(piece) => text.push_str(&piece),
                ModelEvent::Done(_) => break,
                ModelEvent::ReasoningDelta(_) => {}
            }
        }
        text
    })
}

/// LF5: switching models restarts the server, and an idle stop never
/// interrupts a request in flight.
#[test]
fn a_switch_restarts_and_an_idle_stop_never_interrupts_a_request() {
    let setup = Setup::new("lf5");
    setup.settings(r#"{"idle_seconds": 0.2}"#);
    let slow = json!({"text": "one two three four five six seven eight", "chunks": 8, "chunk_delay_ms": 120});
    let a = setup.model("alpha", slow.clone());
    let b = setup.model("beta", json!({}));
    let rt = tokio();
    let runtime = setup.runtime(rt.handle().clone());

    let opened = rt.block_on(runtime.open(a.clone())).unwrap();
    let started = Instant::now();
    let text = chat(&rt, &opened.endpoint);
    let took = started.elapsed();
    println!("a request of {took:?} with idle_seconds 0.2");
    assert!(
        took > Duration::from_millis(500),
        "the request outlasts the idle time"
    );
    assert_eq!(
        text, "one two three four five six seven eight",
        "the whole answer arrived"
    );
    assert_eq!(
        runtime.running().as_deref(),
        Some("alpha"),
        "no idle stop while the lease is held"
    );
    let first_port = port_of(&opened.endpoint);
    let first_pid = runtime.pid().unwrap();
    drop(opened);

    let again = rt.block_on(runtime.open(a.clone())).unwrap();
    assert_eq!(
        runtime.pid(),
        Some(first_pid),
        "the same model reuses the running server"
    );
    drop(again);

    let opened = rt.block_on(runtime.open(b.clone())).unwrap();
    assert_eq!(runtime.running().as_deref(), Some("beta"));
    assert_ne!(
        runtime.pid(),
        Some(first_pid),
        "another model restarts the server"
    );
    let old = rt.block_on(LoopbackHttp::new().unwrap().get(HttpRequest {
        url: format!("http://127.0.0.1:{first_port}/health"),
        bearer: None,
        timeout: Duration::from_secs(2),
    }));
    assert!(old.is_err(), "the first server is gone: {old:?}");
    assert!(key_gone(&launches(&a)[0]));
    drop(opened);
    assert!(
        eventually(Duration::from_secs(5), || runtime.running().is_none()),
        "idle once the last request ended"
    );
    assert!(
        eventually(Duration::from_secs(5), || key_gone(&launches(&b)[0])),
        "the idle stop removes the token (it closes on the blocking pool)"
    );
}

/// X2b (spec §22.6): a `.cmd` wrapper configured as the binary is refused,
/// and a model whose file name holds `cmd.exe` metacharacters launches
/// nothing: no process, no log, no request. The wrapper would exit at once,
/// so if `cmd.exe` ever ran it, the injected `echo.` would print its marker
/// into the server's log.
/// Mutants: the `.exe` rule of `find_binary` dropped (the spawn still
/// refuses); both it and the spawn's batch-file refusal dropped (the marker
/// is printed).
#[test]
fn a_cmd_wrapper_is_refused_and_a_model_name_launches_nothing_extra() {
    let root = Scratch::new("x2b-wrapper");
    let wrapper = root.0.join("run-server.cmd");
    std::fs::write(&wrapper, "@exit /b 0\r\n").unwrap();
    let setup = Setup::with_binary("x2b", &wrapper);
    let model = setup.model("tiny&echo.LATTICE-INJECTED", json!({}));
    let rt = tokio();
    let runtime = setup.runtime(rt.handle().clone());
    let outcome = rt.block_on(runtime.open(model.clone()));
    let logs = setup.paths().logs_dir();
    let printed: Vec<String> = std::fs::read_dir(&logs)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| std::fs::read_to_string(entry.path()).unwrap_or_default())
                .collect()
        })
        .unwrap_or_default();
    println!("outcome {outcome:?}; logs {printed:?}");
    assert!(
        !printed.iter().any(|text| text.contains("LATTICE-INJECTED")),
        "cmd.exe ran the model name as a command: {printed:?}"
    );
    assert_eq!(
        outcome.err(),
        Some(LlamaError::Binary(BinaryProblem::NotExe))
    );
    assert!(launches(&model).is_empty(), "nothing was started");
    assert!(setup.urls().is_empty(), "nothing was asked");
    rt.block_on(runtime.shutdown());
}

/// A listener of this test's own on 127.0.0.1 that answers every request
/// with 200 and `{"status": "ok"}` (so it passes for a ready server's
/// `/health`), and keeps each request's head: its request line and headers.
struct Squatter {
    port: u16,
    seen: Arc<Mutex<Vec<String>>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Squatter {
    fn new() -> Self {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (kept, halt) = (seen.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !halt.load(Ordering::Relaxed) {
                let Ok((mut stream, _)) = listener.accept() else {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                };
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") && head.len() < 64 * 1024 {
                    match stream.read(&mut byte) {
                        Ok(1) => head.push(byte[0]),
                        _ => break,
                    }
                }
                kept.lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&head).into_owned());
                let body = r#"{"status": "ok"}"#;
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        Self {
            port,
            seen,
            stop,
            thread: Some(thread),
        }
    }

    fn seen(&self) -> Vec<String> {
        self.seen.lock().unwrap().clone()
    }
}

impl Drop for Squatter {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// What a Local turn sends once the start succeeded: one streamed chat
/// request with the token, its answer read to the end whatever it is.
fn try_chat(rt: &tokio::runtime::Runtime, endpoint: &ManagedEndpoint) {
    let Ok(built) = models::build_managed(endpoint, None) else {
        return;
    };
    rt.block_on(async {
        let mut stream = built.model.stream(ModelRequest {
            system: "the system prompt".into(),
            input: vec![InputItem::User("a private question".into())],
            tools: Vec::new(),
            settings: ModelSettings::default(),
        });
        while let Some(event) = stream.next().await {
            if matches!(event, Err(_) | Ok(ModelEvent::Done(_))) {
                break;
            }
        }
    });
}

/// LR7a (spec §22.6): a process other than the child that listens on the
/// chosen port, and answers `/health` with 200, fails readiness. Nothing is
/// sent to it: no `/health`, no chat request, no token. The child is
/// stopped and a new port is tried once; here the new port is taken too, so
/// the start fails, saying why. The stub waits 1.5 s before it binds, as
/// llama.cpp loads its backends first, so the squatter has the port first.
/// Mutant: readiness without the owning-process check (`listener_owner` says
/// `Child`).
#[test]
fn a_squatter_on_the_chosen_port_gets_no_request_and_no_token() {
    let setup = Setup::new("lr7a-squat");
    let model = setup.model("tiny", json!({"bind_delay_ms": 1500}));
    let squatter = Squatter::new();
    let rt = tokio();
    let runtime = setup.runtime_on_ports(rt.handle().clone(), vec![squatter.port, squatter.port]);
    let outcome = rt.block_on(runtime.open(model.clone()));
    if let Ok(opened) = &outcome {
        try_chat(&rt, &opened.endpoint);
    }
    let seen = squatter.seen();
    println!("outcome {outcome:?}; the squatter saw {seen:?}");
    assert!(
        !seen
            .iter()
            .any(|head| head.to_ascii_lowercase().contains("authorization")),
        "the token reached the squatter: {seen:?}"
    );
    assert!(
        !seen
            .iter()
            .any(|head| head.contains("/v1/chat/completions")),
        "a chat request reached the squatter: {seen:?}"
    );
    assert!(seen.is_empty(), "nothing at all is sent: {seen:?}");
    assert_eq!(outcome.err(), Some(LlamaError::PortTaken));
    assert_eq!(runtime.running(), None);
    assert_eq!(
        runtime.failed().as_deref(),
        Some(LlamaError::PortTaken.sentence())
    );
    rt.block_on(runtime.shutdown());
}

/// LR7a: when the first port is taken, the start is tried once more on a new
/// one, and the server that answers there is the child: its own process
/// recorded the port it bound.
#[test]
fn a_taken_port_is_left_for_a_new_one() {
    let setup = Setup::new("lr7a-retry");
    let model = setup.model("tiny", json!({"bind_delay_ms": 300}));
    let squatter = Squatter::new();
    let rt = tokio();
    let runtime = setup.runtime_on_ports(rt.handle().clone(), vec![squatter.port]);
    let opened = rt.block_on(runtime.open(model.clone())).unwrap();
    let port = port_of(&opened.endpoint);
    assert_ne!(port, squatter.port);
    assert!(squatter.seen().is_empty(), "{:?}", squatter.seen());
    let pid = runtime.pid().unwrap();
    let ours = launches(&model)
        .into_iter()
        .find(|launch| launch["pid"] == json!(pid))
        .unwrap();
    assert_eq!(ours["bound"], format!("127.0.0.1:{port}"));
    assert!(rt.block_on(runtime.props(&opened.endpoint)).is_ok());
    drop(opened);
    rt.block_on(runtime.shutdown());
}

/// Milliseconds since the Unix epoch, as the stub records its start.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// LR7b (spec §22.6): an idle stop is serialised with starts. The old
/// server's close is held open for 1 s (the runtime's closed hook); a Local
/// turn that arrives once the idle stop has begun waits, and the new server's
/// process starts only after the old one has ended and its token is gone.
/// Mutant: the idle stop hands the close to the blocking pool without the
/// start mutex (the code before LR7b).
#[test]
fn an_open_during_an_idle_stop_waits_for_the_old_server_to_close() {
    let setup = Setup::new("lr7b");
    setup.settings(r#"{"idle_seconds": 0.2}"#);
    let model = setup.model("tiny", json!({}));
    let rt = tokio();
    let closes: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = closes.clone();
    let runtime = setup.runtime_with(rt.handle().clone(), move |config| {
        config.closed_hook = Some(Arc::new(move || {
            std::thread::sleep(Duration::from_secs(1));
            seen.lock().unwrap().push(now_ms());
        }));
    });
    let opened = rt.block_on(runtime.open(model.clone())).unwrap();
    let first_pid = runtime.pid().unwrap();
    let first_key = key_file(&launches(&model)[0]);
    drop(opened);
    // The idle stop has begun once the server has left the slot.
    assert!(eventually(Duration::from_secs(5), || runtime
        .running()
        .is_none()));
    let opened = rt.block_on(runtime.open(model.clone())).unwrap();
    assert!(
        eventually(Duration::from_secs(5), || !closes
            .lock()
            .unwrap()
            .is_empty()),
        "the old server closed"
    );
    let closed_at = closes.lock().unwrap()[0];
    let second = launches(&model)
        .into_iter()
        .find(|launch| launch["pid"] != json!(first_pid))
        .expect("a second launch");
    let started = second["started_ms"].as_u64().unwrap();
    println!(
        "old close finished at {closed_at}, new server started at {started} ({} ms later)",
        started as i64 - closed_at as i64
    );
    assert!(
        started + 20 >= closed_at,
        "the new server started {} ms before the old one had closed",
        closed_at as i64 - started as i64
    );
    assert!(!first_key.exists(), "the old token is gone");
    drop(opened);
    rt.block_on(runtime.shutdown());
}

/// LF1: a hosted endpoint and its key are configured, and no local server can
/// start (no binary; the chosen model is no file; the server exits while it
/// loads). Local and Auto resolve to the managed server or refuse, each with
/// an explicit offline sentence, and nothing the runtime asks, and no client
/// it would build, reaches an address off the loopback.
#[test]
fn with_no_local_server_local_and_auto_refuse_offline_and_nothing_leaves() {
    let hosted = r#"{"version": 1, "endpoints": [
        {"id": "hosted", "label": "Hosted", "base_url": "https://hosted.example.test/v1",
         "model": "big", "api_key_name": "HOSTED_API_KEY", "enabled": true}
    ]}"#;
    let missing_binary = Setup::with_binary(
        "lf1-binary",
        &std::env::temp_dir().join("no-such-llama-server.exe"),
    );
    missing_binary.model("tiny", json!({}));
    missing_binary.choose("tiny");
    let missing_model = Setup::new("lf1-model");
    missing_model.model("tiny", json!({}));
    missing_model.choose("other");
    let exits = Setup::new("lf1-exits");
    exits.model("tiny", json!({"exit_during_load": true, "load_ms": 300}));
    exits.choose("tiny");

    let rt = tokio();
    for (what, setup) in [
        ("no binary", &missing_binary),
        ("no model file", &missing_model),
        ("exits while loading", &exits),
    ] {
        std::fs::write(setup.state.globals.join("model_endpoints.json"), hosted).unwrap();
        let keys = KeyStore::new(setup.env.clone(), &setup.state);
        let configs = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen = configs.clone();
        let factory: ModelFactory =
            Arc::new(move |config: &lattice_agents::ChatCompletionsConfig| {
                seen.lock().unwrap().push(config.base_url.clone());
                None
            });
        let runtime = setup.runtime(rt.handle().clone());
        let view = LocalView::read(setup.env.as_ref(), &setup.state);
        let hosted_ready =
            vocab::chat_choices(setup.env.as_ref(), &setup.state, &keys, false, &view)
                .entries
                .iter()
                .any(|entry| entry.choice.id == "endpoint:hosted" && entry.choice.ready);
        assert!(
            hosted_ready,
            "{what}: the hosted endpoint is ready, and still unused"
        );
        for choice in ["local", "auto"] {
            let resolution = vocab::resolve(
                choice,
                setup.env.as_ref(),
                &setup.state,
                &keys,
                false,
                &view,
            );
            let sentence = match resolution.target {
                Target::Refused(sentence) => sentence,
                Target::Managed(model) => {
                    let opened = rt.block_on(runtime.open(model));
                    match opened {
                        Ok(opened) => {
                            let _ = models::build_managed(&opened.endpoint, Some(&factory));
                            panic!("{what}: {choice}: a server started")
                        }
                        Err(error) => error.sentence().to_owned(),
                    }
                }
                other => panic!("{what}: {choice} resolved to {other:?}"),
            };
            println!("{what}: {choice}: {sentence}");
            assert!(
                sentence.contains("The local model is offline: "),
                "{what}: {choice}: {sentence}"
            );
        }
        assert!(
            configs.lock().unwrap().is_empty(),
            "{what}: no client was built"
        );
        let urls = setup.urls();
        assert!(
            urls.iter().all(|url| is_loopback_url(url)),
            "{what}: {urls:?}"
        );
    }
}

/// LR8′ through the real client and server process: the tool-call probe sent
/// to a server that answers with one streamed call of the echo with `ok`
/// passes; one that answers with prose, other arguments or two calls fails.
#[test]
fn the_tool_call_probe_passes_only_a_well_formed_call_over_the_real_client() {
    let setup = Setup::new("probe");
    let rt = tokio();
    let runtime = setup.runtime(rt.handle().clone());
    let echo = |arguments: &str| json!({"name": "lattice_probe_echo", "arguments": arguments});
    for (name, behaviour, passes) in [
        (
            "good",
            json!({"tool_calls": [echo(r#"{"text": "ok"}"#)]}),
            true,
        ),
        ("prose", json!({"text": "I would call it."}), false),
        (
            "other",
            json!({"tool_calls": [echo(r#"{"text": "no"}"#)]}),
            false,
        ),
        (
            "two",
            json!({"tool_calls": [echo(r#"{"text": "ok"}"#), echo(r#"{"text": "ok"}"#)]}),
            false,
        ),
    ] {
        let model = setup.model(name, behaviour);
        let opened = rt.block_on(runtime.open(model)).unwrap();
        let built = lattice_core::models::build_managed(&opened.endpoint, None).unwrap();
        let outcome = rt.block_on(lattice_core::convo::probe::run(built.model.as_ref()));
        assert_eq!(outcome.passed, passes, "{name}: {}", outcome.reply);
        drop(opened);
    }
    rt.block_on(runtime.shutdown());
}

/// LR8: `/props` read with the token gives the template, the context and
/// vision; tools are assumed only once the tool-call probe is recorded for
/// this binary and model.
#[test]
fn props_and_the_probe_record_decide_the_capabilities() {
    let setup = Setup::new("props");
    let model = setup.model("tiny", json!({"vision": true}));
    let rt = tokio();
    let runtime = setup.runtime(rt.handle().clone());
    let opened = rt.block_on(runtime.open(model.clone())).unwrap();
    let props = rt.block_on(runtime.props(&opened.endpoint)).unwrap();
    assert!(props.chat_template.is_some());
    assert_eq!(props.n_ctx, Some(8192));
    assert_eq!(props.vision, Some(true));
    let caps = managed_caps(None, Some(&props), false);
    assert_eq!(
        (caps.tools, caps.vision, caps.context_tokens),
        (Tri::No, Tri::Yes, Some(8192))
    );
    let paths = setup.paths();
    let sha = probes::binary_sha256(Path::new(STUB)).unwrap();
    let key = probes::probe_key(&sha, &model.path).unwrap();
    std::fs::create_dir_all(&paths.llama_dir).unwrap();
    std::fs::write(
        paths.llama_dir.join(probes::TOOL_PROBES),
        json!({ key.clone(): true }).to_string(),
    )
    .unwrap();
    let passed = probes::recorded(&paths, probes::TOOL_PROBES, &key);
    assert!(passed);
    assert_eq!(managed_caps(None, Some(&props), passed).tools, Tri::Yes);
    let mut stale = opened.endpoint.clone();
    stale.token = "not-the-token".into();
    assert!(
        rt.block_on(runtime.props(&stale)).is_err(),
        "/props refuses another token"
    );
    drop(opened);
    rt.block_on(runtime.shutdown());
}

/// The editor's completions on the real runtime and client, against the stub:
/// a completion asks `/infill` over loopback with the launch's token, and
/// never switches the server from the model a chat is using; nothing about
/// the running server changes when it is refused.
#[test]
fn a_completion_asks_infill_and_never_switches_the_model() {
    use lattice_core::complete::settings::Choice;
    use lattice_core::complete::{Completer, Net, Request};
    use lattice_core::llama::ManagedRuntime;

    let setup = Setup::new("complete");
    let chat_model = setup.model("chat-model", json!({}));
    let coder = setup.model("qwen-coder", json!({"text": "1 + 2;"}));
    let rt = tokio();
    let runtime = Arc::new(setup.runtime(rt.handle().clone()));
    let completer = Completer::new(
        setup.env.clone(),
        setup.state.clone(),
        runtime.clone(),
        Arc::new(Net::new().unwrap()),
    );
    let request = Request { prefix: "let x = ".into(), suffix: "\n".into() };

    let held = rt.block_on(runtime.open(chat_model.clone())).unwrap();
    let pid = runtime.pid().unwrap();
    let refused = rt.block_on(completer.complete(&Choice::Local { model: "qwen-coder".into() }, &request));
    assert_eq!(refused, Err(LlamaError::Serving.sentence().to_owned()));
    let unswitched = rt.block_on(ManagedRuntime::open_unswitched(runtime.as_ref(), coder.clone()));
    assert_eq!(unswitched.err(), Some(LlamaError::Serving));
    assert_eq!(runtime.pid(), Some(pid), "the chat's server was not stopped");
    assert_eq!(runtime.running().as_deref(), Some("chat-model"));
    drop(held);
    rt.block_on(runtime.shutdown());

    let runtime = Arc::new(setup.runtime(rt.handle().clone()));
    let completer = Completer::new(
        setup.env.clone(),
        setup.state.clone(),
        runtime.clone(),
        Arc::new(Net::new().unwrap()),
    );
    let answer = rt.block_on(completer.complete(&Choice::Local { model: "qwen-coder".into() }, &request));
    assert_eq!(answer, Ok(Some("1 + 2;".to_owned())), "with nothing running, the coder starts");
    let mut asked = coder.path.clone().into_os_string();
    asked.push(".stub-infill.json");
    let body: Value = serde_json::from_slice(&std::fs::read(asked).unwrap()).unwrap();
    assert_eq!(body["input_prefix"], "let x = ");
    assert_eq!(body["input_suffix"], "\n");
    let log = std::fs::read_to_string(coder.path.with_file_name("qwen-coder.gguf.stub-requests.log")).unwrap();
    assert!(log.lines().any(|l| l.ends_with("POST /infill token")), "{log}");
    rt.block_on(runtime.shutdown());
}

static UNPARKS: AtomicUsize = AtomicUsize::new(0);

/// CB1 for the runtime: with a server running and idle, the core's runtime
/// sleeps. One idle timer is armed when the last request ends; nothing polls.
/// The count is the runtime's thread unparks over 2 s, after a settle.
#[test]
fn an_idle_server_costs_the_runtime_no_wakeups() {
    let setup = Setup::new("cb1");
    let model = setup.model("tiny", json!({}));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .on_thread_unpark(|| {
            UNPARKS.fetch_add(1, Ordering::Relaxed);
        })
        .build()
        .unwrap();
    let runtime = setup.runtime(rt.handle().clone());
    drop(rt.block_on(runtime.open(model)).unwrap());
    std::thread::sleep(Duration::from_millis(300));
    let before = UNPARKS.load(Ordering::Relaxed);
    std::thread::sleep(Duration::from_secs(2));
    let after = UNPARKS.load(Ordering::Relaxed);
    println!("unparks while idle over 2 s: {}", after - before);
    assert_eq!(
        runtime.running().as_deref(),
        Some("tiny"),
        "the server is still up"
    );
    assert_eq!(after - before, 0);
    rt.block_on(runtime.shutdown());
}
