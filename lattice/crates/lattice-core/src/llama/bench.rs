//! One bounded generation on this machine, measured by llama.cpp's own counters: the native port of
//! the Python runtime's `bench.py` (`measure_generation`) and of `economics.throughput_from_llamacpp`.
//!
//! Python's three rules hold here too, each a refusal rather than a warning: **never automatic** (a caller runs it
//! because a person pressed a button: it loads a model, takes the card and time); **loopback only** (the request goes
//! to the server this function started, on `127.0.0.1`, through [`LoopbackHttp`], which refuses anything else);
//! **bounded and honest** (`n_predict` caps the work, `cache_prompt` is off so the prompt is really processed, and
//! the reading carries the exact prompt, token cap and context, because a rate over 8 tokens and one over 512 are not
//! the same number).
//!
//! The rates come from the server's own `timings` (`predicted_n` tokens in `predicted_ms`, `prompt_n` in
//! `prompt_ms`), which exclude the model's load; the load is timed separately on the wall clock and reported as its
//! own figure, never folded into a rate.
//!
//! Deviation from Python, named: Python measures on the platform's managed server. That server is launched with
//! `-ngl <gpu_layers>` (999 by default: every layer on the card), and a model larger than the card's free memory
//! then fails to load where llama.cpp could have split it. So a measurement runs on a server of its own, started here
//! with the managed server's safeguards (the binary it would use, loopback only, a per-launch token in a key file
//! removed once the server is up, the stripped environment, the Job Object's limits, the listener checked as the
//! child's own) and with `--fit on`, which lets llama.cpp place what fits on the card and keep the rest on the CPU.
//! A binary that does not know `--fit` is started once more with `-ngl 999`, and the reading says which was used.
//! The server is stopped as soon as the answer is in. The caller must not run one while the chat's own server holds
//! the card ([`super::server::LlamaRuntime::running`]).
//!
//! What a reading is not: it is one sample, on one machine, at one context, with one prompt, at whatever the
//! machine's clocks and other work were doing. It says nothing about answer quality, and a second run will differ.

use std::ffi::{OsStr, OsString};
use std::fs::OpenOptions;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lattice_agents::SecretString;
use lattice_sys::process::SpawnRequest;
use serde_json::{Value, json};

use super::files::{LocalModel, find_binary};
use super::server::{
    ListenerOwner, RuntimeConfig, drain, forget_key, healthy, listener_owner, new_token, server_environment, write_key,
};
use crate::net::{LOOPBACK, LoopbackHttp};

/// Enough tokens that the decode rate is not dominated by the first token, few enough that a slow model does not
/// hold the card for a minute (`bench.DEFAULT_MAX_TOKENS`).
pub const DEFAULT_MAX_TOKENS: u32 = 128;
/// A prompt with no domain in it (`bench.DEFAULT_PROMPT`).
pub const DEFAULT_PROMPT: &str = "Write a short paragraph about tide pools.";
/// The context the measuring server is started with.
pub const DEFAULT_CONTEXT: u64 = 4096;
/// The generation's own time limit (`bench.DEFAULT_TIMEOUT`): generous, because a cold machine is slow, but bounded.
pub const GENERATION_TIMEOUT: Duration = Duration::from_secs(600);

/// A cumulative counter read immediately before and after the generation request (and so not across the model's
/// load): the caller's, for example the CPU package's energy counter. `None` when it cannot be read.
#[derive(Clone)]
pub struct Meter(pub Arc<dyn Fn() -> Option<i64> + Send + Sync>);

impl std::fmt::Debug for Meter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Meter")
    }
}

/// What to measure.
#[derive(Clone, Debug)]
pub struct BenchRequest {
    pub model: LocalModel,
    pub prompt: String,
    pub max_tokens: u32,
    pub context: u64,
    pub meter: Option<Meter>,
}

impl BenchRequest {
    /// Python's defaults for `model`.
    pub fn new(model: LocalModel) -> Self {
        Self {
            model,
            prompt: DEFAULT_PROMPT.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            context: DEFAULT_CONTEXT,
            meter: None,
        }
    }
}

/// How llama.cpp was told to place the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// `--fit on`: llama.cpp put on the card what fits there.
    Fit,
    /// `-ngl 999`: every layer on the card (a binary without `--fit`).
    AllLayers,
}

/// The server's own counters for one generation (`timings` in its `/completion` answer).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timings {
    pub prompt_tokens: u64,
    pub prompt_ms: f64,
    pub predicted_tokens: u64,
    pub predicted_ms: f64,
}

impl Timings {
    /// Prefill: prompt tokens per second (`throughput_from_llamacpp`'s `prefill_tokens_per_second`).
    pub fn prefill_per_second(&self) -> Option<f64> {
        rate(self.prompt_tokens, self.prompt_ms)
    }

    /// Decode: generated tokens per second.
    pub fn decode_per_second(&self) -> Option<f64> {
        rate(self.predicted_tokens, self.predicted_ms)
    }
}

fn rate(count: u64, millis: f64) -> Option<f64> {
    (count > 0 && millis > 0.0 && millis.is_finite()).then(|| count as f64 / (millis / 1e3))
}

/// One measurement, with the exact conditions it was taken under.
#[derive(Clone, Debug)]
pub struct BenchReading {
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub context: u64,
    pub placement: Placement,
    /// From the start of the process to the server answering `/health` (the model's load), on the wall clock.
    pub load: Duration,
    /// The generation request on the wall clock (the rates do not use it).
    pub request: Duration,
    /// The server's counters; `None` where its answer carried none usable (UNMEASURED, never a wall-clock stand-in).
    pub timings: Option<Timings>,
    /// The server's answer, verbatim.
    pub payload: Value,
    /// The meter's readings just before and just after the generation request, when one was given and both read.
    pub metered: Option<(i64, i64)>,
}

impl BenchReading {
    /// How the figures were taken (`GenerationReading.method`), so a figure quoted out of context carries its
    /// conditions.
    pub fn method(&self) -> String {
        format!(
            "one generation of up to {} tokens at {} context, rates from the runtime's own timings (which exclude model \
             load time)",
            self.max_tokens, self.context
        )
    }
}

/// Why there is no reading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BenchError {
    /// The token cap was not positive: a measurement of zero tokens has no rate in it.
    NoTokens,
    /// No llama.cpp server binary was found (the managed server's own rule).
    Binary(String),
    /// The launch could not be prepared or started.
    Launch(&'static str),
    /// Another process listened on the port chosen for the server.
    PortTaken,
    /// The server ended before it was ready; the end of its log says why.
    Exited(String),
    /// It was not ready within the start time limit.
    NotReady,
    /// The generation failed.
    Generation(String),
}

impl std::fmt::Display for BenchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchError::NoTokens => {
                f.write_str("the token cap must be positive; a measurement of zero tokens has no rate in it")
            }
            BenchError::Binary(why) => write!(f, "no llama.cpp server to measure with: {why}"),
            BenchError::Launch(why) => write!(f, "the measuring server could not be started: {why}"),
            BenchError::PortTaken => f.write_str("another program took the port chosen for the measuring server"),
            BenchError::Exited(tail) => write!(f, "llama.cpp's server stopped while loading the model: {tail}"),
            BenchError::NotReady => f.write_str("llama.cpp's server did not become ready in time"),
            BenchError::Generation(why) => write!(f, "the runtime did not complete a generation: {why}"),
        }
    }
}

/// The measuring server's command line: the managed server's (LR6) with its own context, one slot, no chat template
/// and the placement asked for.
pub fn command(
    binary: &std::path::Path,
    port: u16,
    request: &BenchRequest,
    key_file: &std::path::Path,
    placement: Placement,
) -> Vec<OsString> {
    let mut argv: Vec<OsString> = vec![binary.as_os_str().to_owned()];
    let mut flag = |name: &str, value: &OsStr| {
        argv.push(name.into());
        argv.push(value.to_owned());
    };
    flag("--host", OsStr::new(LOOPBACK));
    flag("--port", OsStr::new(&port.to_string()));
    flag("-m", request.model.path.as_os_str());
    flag("--alias", OsStr::new(&request.model.name));
    flag("-c", OsStr::new(&request.context.to_string()));
    match placement {
        Placement::Fit => flag("--fit", OsStr::new("on")),
        Placement::AllLayers => flag("-ngl", OsStr::new("999")),
    }
    flag("-np", OsStr::new("1"));
    flag("--api-key-file", key_file.as_os_str());
    argv.push("--no-webui".into());
    argv.push("--offline".into());
    argv
}

/// `throughput_from_llamacpp`'s reading of an answer's `timings`: integer counts (never a boolean) and positive
/// numeric milliseconds, else nothing.
pub fn timings_of(payload: &Value) -> Option<Timings> {
    let timings = payload.get("timings")?;
    let count = |key: &str| match timings.get(key) {
        Some(Value::Number(n)) => n.as_u64(),
        _ => None,
    };
    let millis = |key: &str| match timings.get(key) {
        Some(Value::Number(n)) => n.as_f64(),
        _ => None,
    };
    let found = Timings {
        prompt_tokens: count("prompt_n")?,
        prompt_ms: millis("prompt_ms")?,
        predicted_tokens: count("predicted_n")?,
        predicted_ms: millis("predicted_ms")?,
    };
    (found.prefill_per_second().is_some() || found.decode_per_second().is_some()).then_some(found)
}

/// The request body (`measure_generation`'s, field for field).
pub fn body(request: &BenchRequest) -> Value {
    json!({
        "prompt": request.prompt,
        "n_predict": request.max_tokens,
        "cache_prompt": false,
        "stream": false,
    })
}

/// A started measuring server, stopped and its token removed when dropped.
struct Measuring {
    child: Arc<lattice_sys::process::Child>,
    key_dir: std::path::PathBuf,
    key_file: std::path::PathBuf,
    stop_timeout: Duration,
}

impl Drop for Measuring {
    fn drop(&mut self) {
        let _ = self.child.kill_tree();
        let _ = self.child.wait(Some(self.stop_timeout));
        forget_key(&self.key_file, &self.key_dir);
    }
}

/// Run one measurement: start a server for the model, wait for it, generate once, stop it. Blocking work (files,
/// `CreateProcess`) happens on the calling thread; call from a blocking-capable context.
pub async fn measure(config: &RuntimeConfig, request: BenchRequest) -> Result<BenchReading, BenchError> {
    if request.max_tokens == 0 {
        return Err(BenchError::NoTokens);
    }
    match run(config, &request, Placement::Fit).await {
        Err(BenchError::Exited(tail)) if tail.contains("--fit") => run(config, &request, Placement::AllLayers).await,
        other => other,
    }
}

async fn run(config: &RuntimeConfig, request: &BenchRequest, placement: Placement) -> Result<BenchReading, BenchError> {
    let binary = find_binary(&config.paths).map_err(|why| BenchError::Binary(format!("{why:?}")))?;
    let port = (config.ports)().map_err(|_| BenchError::Launch("no free loopback port"))?;
    let token: SecretString = new_token();
    let (key_dir, key_file) = write_key(&config.temp_roots, &token).map_err(|_| BenchError::Launch("no token file"))?;
    let log_path = config.paths.logs_dir().join(format!("{}.bench.log", request.model.name));
    let opened = std::fs::create_dir_all(config.paths.logs_dir())
        .and_then(|()| OpenOptions::new().create(true).write(true).truncate(true).open(&log_path));
    let Ok(log) = opened else {
        forget_key(&key_file, &key_dir);
        return Err(BenchError::Launch("its log could not be opened"));
    };
    let argv = command(&binary, port, request, &key_file, placement);
    let environment = server_environment(config.env.as_ref(), &config.globals);
    let cwd = binary.parent().unwrap_or(&binary).to_path_buf();
    let started = Instant::now();
    let mut child = match lattice_sys::process::spawn(&SpawnRequest {
        program: &binary,
        argv: &argv,
        cwd: &cwd,
        env: &environment,
        limits: config.limits,
    }) {
        Ok(child) => child,
        Err(_) => {
            forget_key(&key_file, &key_dir);
            return Err(BenchError::Launch("the process did not start"));
        }
    };
    let (stdout, stderr) = (child.take_stdout(), child.take_stderr());
    let log = Arc::new(Mutex::new(log));
    if let Some(stdout) = stdout {
        drain(stdout, log.clone(), None);
    }
    if let Some(stderr) = stderr {
        drain(stderr, log, None);
    }
    let server = Measuring { child: Arc::new(child), key_dir, key_file, stop_timeout: config.stop_timeout };
    let base_url = format!("http://{LOOPBACK}:{port}");
    let pid = server.child.pid();
    let deadline = Instant::now() + config.start_timeout;
    loop {
        let listener = listener_owner(port, pid);
        if listener == ListenerOwner::Other {
            return Err(BenchError::PortTaken);
        }
        if !matches!(server.child.wait(Some(Duration::ZERO)), Ok(None)) {
            // Let the log's last lines land before reading them.
            tokio::time::sleep(Duration::from_millis(200)).await;
            return Err(BenchError::Exited(log_tail(&log_path)));
        }
        if listener == ListenerOwner::Child
            && healthy(config.http.as_ref(), &base_url, config.health_timeout).await
            && listener_owner(port, pid) == ListenerOwner::Child
        {
            // llama.cpp has read the key file by now: it goes at once (LR6a).
            forget_key(&server.key_file, &server.key_dir);
            break;
        }
        if Instant::now() >= deadline {
            return Err(BenchError::NotReady);
        }
        tokio::time::sleep(config.poll_gap).await;
    }
    let load = started.elapsed();
    let client = LoopbackHttp::new().map_err(|_| BenchError::Generation("no loopback client".into()))?;
    let read = || request.meter.as_ref().and_then(|meter| (meter.0)());
    let before = read();
    let asked = Instant::now();
    let answer = client
        .post_json(&format!("{base_url}/completion"), Some(&token), &body(request), GENERATION_TIMEOUT)
        .await
        .map_err(|why| BenchError::Generation(format!("{why:?}")))?;
    let elapsed = asked.elapsed();
    let after = read();
    drop(server);
    if answer.status != 200 {
        return Err(BenchError::Generation(format!("the server answered {}", answer.status)));
    }
    let payload: Value = serde_json::from_slice(&answer.body)
        .ok()
        .filter(Value::is_object)
        .ok_or_else(|| BenchError::Generation("the runtime's answer was not an object".into()))?;
    Ok(BenchReading {
        model: request.model.name.clone(),
        prompt: request.prompt.clone(),
        max_tokens: request.max_tokens,
        context: request.context,
        placement,
        load,
        request: elapsed,
        timings: timings_of(&payload),
        payload,
        metered: before.zip(after),
    })
}

/// The last lines of the measuring server's log, for a refusal that says why it stopped.
fn log_tail(path: &std::path::Path) -> String {
    let text = std::fs::read(path).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let tail = lines[lines.len().saturating_sub(4)..].join(" | ");
    if tail.is_empty() {
        return "its log is empty".to_string();
    }
    let kept: String = tail.chars().take(600).collect();
    if kept.len() < tail.len() { format!("{kept}…") } else { kept }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> LocalModel {
        LocalModel { name: "tiny".into(), path: std::path::PathBuf::from("C:/models/tiny.gguf"), size: 1 }
    }

    #[test]
    fn the_body_is_pythons_and_the_command_keeps_the_managed_safeguards() {
        let request = BenchRequest::new(model());
        assert_eq!(
            body(&request),
            json!({"prompt": DEFAULT_PROMPT, "n_predict": 128, "cache_prompt": false, "stream": false})
        );
        let argv: Vec<String> = command(
            std::path::Path::new("C:/llama/llama-server.exe"),
            5000,
            &request,
            std::path::Path::new("C:/t/key"),
            Placement::Fit,
        )
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
        let pairs = argv.windows(2).map(|w| (w[0].as_str(), w[1].as_str())).collect::<Vec<_>>();
        for pair in [("--host", "127.0.0.1"), ("--port", "5000"), ("-c", "4096"), ("--fit", "on"), ("-np", "1")] {
            assert!(pairs.contains(&pair), "{pair:?} in {argv:?}");
        }
        assert!(pairs.contains(&("--api-key-file", "C:/t/key")) && argv.contains(&"--offline".to_string()));
        assert!(!argv.iter().any(|a| a == "-ngl"), "fit leaves the layer count to llama.cpp");
        let all = command(
            std::path::Path::new("C:/llama/llama-server.exe"),
            5000,
            &request,
            std::path::Path::new("C:/t/key"),
            Placement::AllLayers,
        );
        assert!(all.windows(2).any(|w| w[0] == "-ngl" && w[1] == "999") && !all.iter().any(|a| a == "--fit"));
    }

    /// Run by hand: a real measurement of the GGUF file `LATTICE_BENCH_MODEL` names, with the server
    /// `ALELYON_LLAMA_SERVER` names (or the managed install). It loads a model and takes the card.
    #[test]
    #[ignore = "loads a real model on this machine's card: run by hand with LATTICE_BENCH_MODEL set"]
    fn a_real_model_is_measured() {
        let path = std::path::PathBuf::from(std::env::var("LATTICE_BENCH_MODEL").expect("LATTICE_BENCH_MODEL"));
        let size = std::fs::metadata(&path).unwrap().len();
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let env: std::sync::Arc<dyn crate::env::Env> = std::sync::Arc::new(crate::env::ProcessEnv);
        let state = crate::state::resolve();
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
        let reading = rt.block_on(async {
            let config = RuntimeConfig::new(env, &state).unwrap();
            measure(&config, BenchRequest::new(LocalModel { name, path, size })).await
        });
        let reading = reading.unwrap_or_else(|why| panic!("{why}"));
        let timings = reading.timings.expect("the server's own timings");
        println!(
            "{} ({:?}): load {:.2} s; prefill {:?} tok/s over {} tokens; decode {:?} tok/s over {} tokens; request {:.2} s",
            reading.model,
            reading.placement,
            reading.load.as_secs_f64(),
            timings.prefill_per_second(),
            timings.prompt_tokens,
            timings.decode_per_second(),
            timings.predicted_tokens,
            reading.request.as_secs_f64()
        );
        println!("{}", reading.method());
    }

    #[test]
    fn rates_come_from_the_servers_counters_and_nothing_else() {
        let answer = json!({"timings": {"prompt_n": 12, "prompt_ms": 20.0, "predicted_n": 128, "predicted_ms": 1450}});
        let timings = timings_of(&answer).unwrap();
        assert_eq!(timings.prefill_per_second(), Some(600.0));
        assert!((timings.decode_per_second().unwrap() - 128.0 / 1.45).abs() < 1e-9);
        for bad in [
            json!({}),
            json!({"timings": "fast"}),
            json!({"timings": {"prompt_n": true, "prompt_ms": 1, "predicted_n": 1, "predicted_ms": 1}}),
            json!({"timings": {"prompt_n": 0, "prompt_ms": 0, "predicted_n": 0, "predicted_ms": 0}}),
            json!({"timings": {"prompt_n": 1.5, "prompt_ms": 1, "predicted_n": 1, "predicted_ms": 1}}),
        ] {
            assert_eq!(timings_of(&bad), None, "{bad}");
        }
        // One rate without the other is kept, its missing rate UNMEASURED.
        let half = timings_of(&json!({"timings": {"prompt_n": 0, "prompt_ms": 0, "predicted_n": 5, "predicted_ms": 50}}));
        let half = half.unwrap();
        assert_eq!((half.prefill_per_second(), half.decode_per_second()), (None, Some(100.0)));
    }
}
