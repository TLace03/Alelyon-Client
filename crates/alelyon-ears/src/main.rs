//! `angel-ears`: the ears from the command line.
//!
//!     angel-ears serve [--port 8186] [--attach <host:port> | --model <path> --whisper-exe <path> --whisper-port 8187 [--cpu] [--threads N]] [--language <code|auto>]
//!     angel-ears transcribe <file> [--format txt|srt|vtt|json] [--out <path>] [--server <host:port>] [--language <code|auto>]
//!     angel-ears listen [--source mic|pc] [--device <name>] [--seconds N] [--start-rms X] [--server <host:port>] [--language <code|auto>]
//!     angel-ears devices
//!
//! `serve` is the service the window and Sinai's hearing connect to (`server`). It starts its own whisper-server
//! with the speech model unless `--attach` names one already running; where it looks for both, and what it says when
//! one is missing, is `setup` (the model `~/.alelyon/angel/models/ggml-large-v3-turbo-q5_0.bin`, the program beside
//! this one). `transcribe` reads WAV and anything Windows can decode (MP3, M4A/AAC, WMA, FLAC); progress
//! goes to stderr, the transcript to stdout or `--out`. `listen` prints live captions from the microphone or
//! from what the computer is playing until Ctrl+C (or for `--seconds`), from the default device or the first
//! whose name contains `--device`. Both use a whisper-server at `--server` (default 127.0.0.1:8187, the one
//! `serve` starts). The ports sit clear of 8178-8182, which Sinai's other local services hold (its speech, mind,
//! face, eyes and embeddings).

use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use alelyon_ears::files;
use alelyon_ears::whisper::WhisperServer;

/// Options that take no value.
const FLAGS: &[&str] = &["cpu"];

struct Args {
    command: String,
    positional: Vec<String>,
    options: Vec<(String, String)>,
}

impl Args {
    fn parse() -> Result<Args, String> {
        let mut raw = std::env::args().skip(1);
        let command = raw.next().ok_or_else(usage)?;
        let mut positional = Vec::new();
        let mut options = Vec::new();
        while let Some(a) = raw.next() {
            if let Some(name) = a.strip_prefix("--") {
                if FLAGS.contains(&name) {
                    options.push((name.to_string(), "true".to_string()));
                    continue;
                }
                let value = raw.next().ok_or_else(|| format!("--{name} needs a value"))?;
                options.push((name.to_string(), value));
            } else {
                positional.push(a);
            }
        }
        Ok(Args { command, positional, options })
    }

    fn opt(&self, name: &str) -> Option<&str> {
        self.options.iter().rev().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    fn flag(&self, name: &str) -> bool {
        self.opt(name) == Some("true")
    }
}

fn usage() -> String {
    [
        "usage: angel-ears serve [--port 8186] [--attach <host:port> | --model <path> --whisper-exe <path> --whisper-port 8187 [--cpu] [--threads N]] [--language <code|auto>]",
        "       angel-ears transcribe <file> [--format txt|srt|vtt|json] [--out <path>] [--server <host:port>] [--language <code|auto>]",
        "       angel-ears listen [--source mic|pc] [--device <name>] [--seconds N] [--start-rms X] [--server <host:port>] [--language <code|auto>]",
        "       angel-ears devices",
    ]
    .join("\n")
}

fn recogniser(args: &Args) -> Result<WhisperServer, String> {
    let addr: SocketAddr = args.opt("server").unwrap_or("127.0.0.1:8187").parse().map_err(|e| format!("--server: {e}"))?;
    let mut server = WhisperServer::new(addr);
    if let Some(lang) = args.opt("language") {
        server.language = Some(lang.to_string());
    }
    if let Some(secs) = args.opt("timeout") {
        server.timeout = Duration::from_secs_f64(secs.parse().map_err(|e| format!("--timeout: {e}"))?);
    }
    if !server.alive() {
        return Err(format!("no whisper-server answers at {addr}; run `angel-ears serve`, or pass --server"));
    }
    Ok(server)
}

fn transcribe(args: &Args) -> Result<(), String> {
    let path = PathBuf::from(args.positional.first().ok_or_else(usage)?);
    let format = args.opt("format").unwrap_or("txt");
    if !["txt", "srt", "vtt", "json"].contains(&format) {
        return Err(format!("--format must be txt, srt, vtt or json, not {format}"));
    }
    let server = recogniser(args)?;
    eprintln!("reading {}", path.display());
    let audio = alelyon_ears::server::load_16k(&path)?;
    let seconds = audio.len() as f64 / f64::from(alelyon_ears::stream::RATE);
    eprintln!("{seconds:.1} s of audio; transcribing with whisper-server at {}", server.addr);
    let began = std::time::Instant::now();
    let mut last_shown = -1i64;
    let result = files::transcribe(&audio, server, None, |f| {
        let pct = (f * 100.0).floor() as i64;
        if pct != last_shown {
            last_shown = pct;
            eprint!("\r  {pct:3}%");
            let _ = std::io::stderr().flush();
        }
        true
    });
    eprintln!("\r  done in {:.1} s ({} lines, {} readings failed)", began.elapsed().as_secs_f64(), result.lines.len(), result.errors.len());
    for e in &result.errors {
        eprintln!("  failed: {e}");
    }
    let text = match format {
        "srt" => files::to_srt(&result),
        "vtt" => files::to_vtt(&result),
        "json" => serde_json::to_string_pretty(&files::to_json(&result, &path.display().to_string())).unwrap_or_default(),
        _ => files::to_text(&result),
    };
    match args.opt("out") {
        Some(out) => std::fs::write(out, text).map_err(|e| format!("{out}: {e}"))?,
        None => print!("{text}"),
    }
    Ok(())
}

fn listen(args: &Args) -> Result<(), String> {
    use alelyon_ears::capture::{Capture, Endpoint};
    use alelyon_ears::resample;
    use alelyon_ears::stream::{Config, Event, Source, Transcriber, RATE};

    let source = match args.opt("source").unwrap_or("mic") {
        "mic" => Source::Mic,
        "pc" => Source::Pc,
        other => return Err(format!("--source must be mic or pc, not {other}")),
    };
    let limit = args.opt("seconds").map(|s| s.parse::<f64>().map_err(|e| format!("--seconds: {e}"))).transpose()?;
    // The live floors (stream::LIVE_START_RMS) unless the run names its own; continuing keeps the detector's ratio.
    let start_rms = match args.opt("start-rms") {
        Some(s) => s.parse::<f32>().map_err(|e| format!("--start-rms: {e}"))?,
        None => alelyon_ears::stream::LIVE_START_RMS,
    };
    let continue_rms = start_rms * alelyon_ears::vad::CONTINUE_RMS / alelyon_ears::vad::START_RMS;
    let server = recogniser(args)?;
    let endpoint = if source == Source::Mic { Endpoint::Microphone } else { Endpoint::Loopback };
    let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let capture = Capture::start_on(endpoint, args.opt("device"), tx)?;
    match limit {
        Some(s) => eprintln!("ON AIR: listening to {} ({}) for {s} s.", capture.device_name(), source.name()),
        None => eprintln!("ON AIR: listening to {} ({}). Ctrl+C stops.", capture.device_name(), source.name()),
    }
    let mut transcriber = Transcriber::new(server, Config::live(source));
    transcriber.set_floors(start_rms, continue_rms);
    let mut resampler = resample::Resampler::new(capture.rate(), RATE);
    let channels = capture.channels();
    let began = std::time::Instant::now();
    // What came in, for the summary at the end: a silent result is then explained, not a mystery.
    let (mut heard, mut levels, mut starts) = (0usize, Vec::<f32>::new(), 0usize);
    // On a console the words in progress redraw one line; into a file each reading is a line of its own, stamped
    // with the seconds since the start, so a log shows how soon the words came.
    let console = std::io::IsTerminal::is_terminal(&std::io::stderr());
    let clear = if console { "\r\x1b[2K" } else { "" };
    let show = |event: Event| match event {
        Event::SpeechStart { .. } if console => eprint!("{clear}  ..."),
        Event::SpeechStart { at, .. } => eprintln!("[{:7.1}s] speech at {at:.1}", began.elapsed().as_secs_f64()),
        Event::Partial { stable, settling, .. } if console => eprint!("{clear}  {stable} \x1b[2m{settling}\x1b[0m"),
        Event::Partial { stable, settling, .. } => {
            eprintln!("[{:7.1}s] ~ {stable} | {settling}", began.elapsed().as_secs_f64());
        }
        // finals go to stdout, so a run can be kept as a transcript
        Event::Final { text, start, .. } => {
            eprint!("{clear}");
            println!("[{start:7.1}] {text}");
        }
        Event::Dropped { .. } => eprint!("{clear}"),
        Event::Error { stage, message, .. } => eprintln!("{clear}  ({stage} reading failed: {message})"),
    };
    loop {
        if limit.is_some_and(|s| began.elapsed().as_secs_f64() >= s) {
            break;
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(block) => {
                let mono = resample::downmix(&block, channels);
                let audio = resampler.process(&mono);
                heard += audio.len();
                levels.extend(audio.chunks(alelyon_ears::vad::FRAME_SAMPLES).map(alelyon_ears::vad::rms));
                for event in transcriber.push(&audio) {
                    if matches!(event, Event::SpeechStart { .. }) {
                        starts += 1;
                    }
                    show(event);
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    drop(capture);
    levels.sort_by(f32::total_cmp);
    let at = |q: f64| levels.get(((levels.len().max(1) - 1) as f64 * q).round() as usize).copied().unwrap_or(0.0);
    let (typical, loudest) = (at(0.5), at(1.0));
    eprintln!(
        "{clear}OFF AIR. Heard {:.1} s; level (30 ms RMS) typical {typical:.4}, 90th {:.4}, 99th {:.4}, loudest \
         {loudest:.4}; speech must pass {start_rms:.4}; speech began {starts} time(s).",
        heard as f64 / f64::from(RATE),
        at(0.9),
        at(0.99)
    );
    if heard > 0 && loudest == 0.0 {
        eprintln!("The device sent only digital silence: it is probably muted, switched off or not connected.");
    }
    for event in transcriber.finish() {
        show(event);
    }
    Ok(())
}

fn home() -> Result<PathBuf, String> {
    std::env::var_os("USERPROFILE").map(PathBuf::from).ok_or_else(|| "USERPROFILE is not set".to_string())
}

fn serve(args: &Args) -> Result<(), String> {
    use alelyon_ears::recognizer::{RecognizerProcess, Spec};
    use alelyon_ears::server::{self, Service, Settings};
    use alelyon_ears::setup;

    let port: u16 = args.opt("port").unwrap_or("8186").parse().map_err(|e| format!("--port: {e}"))?;
    let bind: SocketAddr = SocketAddr::from(([127, 0, 0, 1], port));
    let language = Some(args.opt("language").unwrap_or("auto").to_string());
    let home = home()?;
    let (recognizer, model, process) = match args.opt("attach") {
        Some(addr) => {
            let addr: SocketAddr = addr.parse().map_err(|e| format!("--attach: {e}"))?;
            if !WhisperServer::new(addr).alive() {
                return Err(format!("no whisper-server answers at {addr}"));
            }
            (addr, format!("whisper-server at {addr}"), None)
        }
        None => {
            let mut given = setup::Given::from_env(None);
            given.model = args.opt("model").map(PathBuf::from);
            given.server = args.opt("whisper-exe").map(PathBuf::from);
            given.home = Some(home.clone());
            let missing = given.missing(|p| p.is_file());
            if !missing.is_empty() {
                return Err(missing.iter().map(setup::Missing::explain).collect::<Vec<_>>().join("\n"));
            }
            let exe = given.server_path(|p| p.is_file()).ok_or("the recogniser program's place is unknown")?;
            let model = given.model_path().ok_or("the speech model's place is unknown")?;
            let spec = Spec {
                exe,
                model: model.clone(),
                port: args.opt("whisper-port").unwrap_or("8187").parse().map_err(|e| format!("--whisper-port: {e}"))?,
                cpu_only: args.flag("cpu"),
                threads: args.opt("threads").map(|t| t.parse().map_err(|e| format!("--threads: {e}"))).transpose()?,
                log: home.join(".alelyon").join("angel").join("logs").join("ears-whisper.log"),
            };
            let name = model.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            eprintln!("[ears] starting whisper-server with {name}{} ...", if spec.cpu_only { " on the processor" } else { "" });
            let p = RecognizerProcess::start(&spec, Duration::from_secs(180))?;
            (p.addr, name, Some(p))
        }
    };
    let listener = std::net::TcpListener::bind(bind).map_err(|e| format!("cannot listen on {bind} (is another ears service running?): {e}"))?;
    let token = server::new_token()?;
    let file = server::connection_file().ok_or("USERPROFILE is not set")?;
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let details = serde_json::json!({"url": format!("ws://{bind}/"), "port": port, "token": token, "pid": std::process::id()});
    std::fs::write(&file, details.to_string()).map_err(|e| format!("{}: {e}", file.display()))?;
    let svc = Service::new(Settings {
        bind,
        recognizer,
        language,
        // The recogniser prompt Sinai's hearing has always used, so the wake word is spelled as Sinai rather than
        // heard as "Sinay".
        mic_prompt: Some("Hey Sinai, can you help me? Thank you, Sinai.".to_string()),
        token,
        model,
    });
    if let Some(mut process) = process {
        // The recogniser is watched: if it exits, every client is told, rather than readings failing quietly.
        let watched = std::sync::Arc::clone(&svc);
        std::thread::Builder::new()
            .name("ears-recognizer-watch".into())
            .spawn(move || {
                while process.running() {
                    std::thread::sleep(Duration::from_secs(2));
                }
                watched.set_recognizer_state("down");
                eprintln!("[ears] whisper-server exited; see ~/.alelyon/angel/logs/ears-whisper.log");
            })
            .map_err(|e| e.to_string())?;
    }
    eprintln!("[ears] listening on ws://{bind}/ (connection details in {})", file.display());
    svc.serve(listener)
}

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let result = match args.command.as_str() {
        "serve" => serve(&args),
        "transcribe" => transcribe(&args),
        "listen" => listen(&args),
        "devices" => alelyon_ears::capture::print_devices(),
        _ => Err(usage()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("angel-ears: {e}");
            ExitCode::FAILURE
        }
    }
}
