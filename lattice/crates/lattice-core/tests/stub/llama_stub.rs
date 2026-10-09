//! A stand-in for llama.cpp's `llama-server`, for the local runtime's tests
//! (the chat core's spec §22.2, LF1–LF5). Test tooling only: it is a
//! binary of this package so the tests can find it through
//! `CARGO_BIN_EXE_lattice-llama-stub`; nothing ships it, and it never runs a
//! model or touches a GPU.
//!
//! It takes the real server's command line (`--host`, `--port`, `-m`,
//! `--alias`, `-c`, `-ngl`, `-np`, `--api-key-file`, `--no-webui`,
//! `--offline`, `--jinja`), reads its token from the key file, and serves on
//! the host and port it is given:
//! - `GET /health`, without a token (as llama.cpp serves it): 503 while it
//!   "loads", then 200;
//! - `GET /props`, `GET /v1/models` and `POST /v1/chat/completions` with
//!   `Authorization: Bearer <token>` only; anything else is 401.
//!
//! What it records, next to the model file: `<model>.stub-<pid>.json`
//! (its argv, its whole environment and the wall-clock millisecond it
//! started, once, for LF3 and LR7b; never the token) and
//! `<model>.stub-requests.log` (one line per request: pid, method, path and
//! whether the token matched). What it does is read from `<model>.stub.json`
//! when that exists: `load_ms`, `exit_during_load`, `exit_after_ms`,
//! `chunks`, `chunk_delay_ms`, `text`, `template` (null for none),
//! `vision`, `bind_delay_ms` (how long it waits before it binds its port,
//! as llama.cpp loads its backends first) and `tool_calls` (a list of
//! `{"name", "arguments"}`: a request that offers tools is answered with
//! them, streamed as llama.cpp streams tool calls, else with `text`).
//!
//! It exits when told to by its behaviour file, or when its parent's Job
//! Object ends it.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

struct Config {
    host: String,
    port: u16,
    model: PathBuf,
    alias: String,
    ctx: String,
    key_file: PathBuf,
}

fn parse(argv: &[String]) -> Config {
    let value = |flag: &str| -> String {
        argv.iter()
            .position(|arg| arg == flag)
            .and_then(|at| argv.get(at + 1))
            .cloned()
            .unwrap_or_default()
    };
    Config {
        host: value("--host"),
        port: value("--port").parse().unwrap_or(0),
        model: PathBuf::from(value("-m")),
        alias: value("--alias"),
        ctx: value("-c"),
        key_file: PathBuf::from(value("--api-key-file")),
    }
}

fn sibling(model: &Path, suffix: &str) -> PathBuf {
    let mut name = model.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    model.with_file_name(name)
}

struct Behaviour {
    load: Duration,
    exit_during_load: bool,
    exit_after: Option<Duration>,
    chunks: usize,
    chunk_delay: Duration,
    text: String,
    template: Option<String>,
    vision: bool,
    bind_delay: Duration,
    tool_calls: Vec<(String, String)>,
}

fn behaviour(model: &Path) -> Behaviour {
    let raw: Value = std::fs::read(sibling(model, ".stub.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_else(|| json!({}));
    let ms = |key: &str, default: u64| Duration::from_millis(raw[key].as_u64().unwrap_or(default));
    Behaviour {
        load: ms("load_ms", 150),
        exit_during_load: raw["exit_during_load"].as_bool().unwrap_or(false),
        exit_after: raw["exit_after_ms"].as_u64().map(Duration::from_millis),
        chunks: raw["chunks"].as_u64().unwrap_or(3) as usize,
        chunk_delay: ms("chunk_delay_ms", 5),
        text: raw["text"].as_str().unwrap_or("stub answer").to_owned(),
        template: match raw.get("template") {
            Some(Value::Null) => None,
            Some(Value::String(text)) => Some(text.clone()),
            _ => Some("{% for message in messages %}{{ message.content }}{% endfor %}".to_owned()),
        },
        vision: raw["vision"].as_bool().unwrap_or(false),
        bind_delay: ms("bind_delay_ms", 0),
        tool_calls: raw["tool_calls"]
            .as_array()
            .map(|calls| {
                calls
                    .iter()
                    .map(|call| {
                        (
                            call["name"].as_str().unwrap_or("").to_owned(),
                            call["arguments"].as_str().unwrap_or("").to_owned(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn append(path: &Path, line: &str) {
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(line.as_bytes());
    }
}

struct Request {
    method: String,
    path: String,
    authorization: Option<String>,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> Option<Request> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    let end = loop {
        if let Some(at) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            break at;
        }
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
    };
    let head = String::from_utf8_lossy(&buffer[..end]).into_owned();
    let mut lines = head.split("\r\n");
    let mut first = lines.next()?.split(' ');
    let method = first.next()?.to_owned();
    let path = first.next()?.to_owned();
    let mut length = 0usize;
    let mut authorization = None;
    for line in lines {
        if let Some((key, value)) = line.split_once(':') {
            let value = value.trim().to_owned();
            if key.trim().eq_ignore_ascii_case("content-length") {
                length = value.parse().unwrap_or(0);
            } else if key.trim().eq_ignore_ascii_case("authorization") {
                authorization = Some(value);
            }
        }
    }
    let mut body = buffer[end + 4..].to_vec();
    while body.len() < length {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    Some(Request {
        method,
        path,
        authorization,
        body,
    })
}

fn respond(stream: &mut TcpStream, code: u16, body: &Value) {
    let text = body.to_string();
    let reason = match code {
        200 => "OK",
        401 => "Unauthorized",
        404 => "Not Found",
        _ => "Service Unavailable",
    };
    let _ = stream.write_all(
        format!(
            "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
            text.len()
        )
        .as_bytes(),
    );
}

struct Server {
    config: Config,
    behaviour: Behaviour,
    token: String,
    started: Instant,
    log: PathBuf,
}

fn chunk(text: &str) -> String {
    let piece = json!({"id": "stub", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]});
    format!("data: {piece}\n\n")
}

/// The behaviour's tool calls as llama.cpp answers them: one message with
/// `tool_calls`, or, streamed, each call's name and then its arguments in two
/// pieces, then `finish_reason: "tool_calls"`.
fn answer_with_calls(stream: &mut TcpStream, calls: &[(String, String)], streamed: bool) {
    if !streamed {
        let tool_calls: Vec<Value> = calls
            .iter()
            .enumerate()
            .map(|(index, (name, arguments))| {
                json!({"id": format!("call_{index}"), "type": "function",
                    "function": {"name": name, "arguments": arguments}})
            })
            .collect();
        respond(
            stream,
            200,
            &json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": null,
                "tool_calls": tool_calls}, "finish_reason": "tool_calls"}]}),
        );
        return;
    }
    let _ = stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
    );
    for (index, (name, arguments)) in calls.iter().enumerate() {
        let (first, rest) = arguments.split_at(arguments.len() / 2);
        let pieces = [
            json!({"index": index, "id": format!("call_{index}"), "type": "function",
                "function": {"name": name, "arguments": first}}),
            json!({"index": index, "function": {"arguments": rest}}),
        ];
        for piece in pieces {
            let delta = json!({"id": "stub", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"tool_calls": [piece]}, "finish_reason": null}]});
            let _ = stream.write_all(format!("data: {delta}\n\n").as_bytes());
        }
    }
    let end = json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3}});
    let _ = stream.write_all(format!("data: {end}\n\ndata: [DONE]\n\n").as_bytes());
    let _ = stream.flush();
}

fn serve(server: &Server, mut stream: TcpStream) {
    let Some(request) = read_request(&mut stream) else {
        return;
    };
    let authorized = request.authorization.as_deref() == Some(&format!("Bearer {}", server.token));
    append(
        &server.log,
        &format!(
            "{} {} {} {}\n",
            std::process::id(),
            request.method,
            request.path,
            if authorized { "token" } else { "no-token" }
        ),
    );
    let ready = server.started.elapsed() >= server.behaviour.load;
    if request.path == "/health" {
        if ready {
            respond(&mut stream, 200, &json!({"status": "ok"}));
        } else {
            respond(
                &mut stream,
                503,
                &json!({"error": {"message": "Loading model"}}),
            );
        }
        return;
    }
    if !authorized {
        respond(
            &mut stream,
            401,
            &json!({"error": {"message": "Invalid API Key"}}),
        );
        return;
    }
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/props") => {
            let mut props = json!({
                "default_generation_settings": {"n_ctx": server.config.ctx.parse::<u64>().unwrap_or(0)},
                "total_slots": 1,
                "modalities": {"vision": server.behaviour.vision},
                "build_info": "stub",
            });
            if let Some(template) = &server.behaviour.template {
                props["chat_template"] = json!(template);
            }
            respond(&mut stream, 200, &props);
        }
        ("GET", "/v1/models") => {
            respond(
                &mut stream,
                200,
                &json!({"object": "list", "data": [{"id": server.config.alias, "object": "model"}]}),
            );
        }
        ("POST", "/v1/chat/completions") => {
            let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
            let text = server.behaviour.text.clone();
            let offers_tools = body["tools"]
                .as_array()
                .is_some_and(|tools| !tools.is_empty());
            if offers_tools && !server.behaviour.tool_calls.is_empty() {
                answer_with_calls(
                    &mut stream,
                    &server.behaviour.tool_calls,
                    body["stream"].as_bool() == Some(true),
                );
                return;
            }
            if body["stream"].as_bool() != Some(true) {
                respond(
                    &mut stream,
                    200,
                    &json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop"}]}),
                );
                return;
            }
            let _ = stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
            );
            let pieces = server.behaviour.chunks.max(1);
            let size = text.chars().count().div_ceil(pieces).max(1);
            let chars: Vec<char> = text.chars().collect();
            for part in chars.chunks(size) {
                let piece: String = part.iter().collect();
                if stream.write_all(chunk(&piece).as_bytes()).is_err() {
                    return;
                }
                let _ = stream.flush();
                std::thread::sleep(server.behaviour.chunk_delay);
            }
            let end = json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3}});
            let _ = stream.write_all(format!("data: {end}\n\ndata: [DONE]\n\n").as_bytes());
            let _ = stream.flush();
        }
        _ => respond(
            &mut stream,
            404,
            &json!({"error": {"message": "not found"}}),
        ),
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let config = parse(&argv);
    let env: BTreeMap<String, String> = std::env::vars_os()
        .map(|(name, value)| {
            (
                name.to_string_lossy().into_owned(),
                value.to_string_lossy().into_owned(),
            )
        })
        .collect();
    let token = std::fs::read_to_string(&config.key_file)
        .map(|text| text.trim().to_owned())
        .unwrap_or_default();
    let behaviour = behaviour(&config.model);
    let started_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_millis() as u64)
        .unwrap_or(0);
    let mut record = json!({"pid": std::process::id(), "argv": argv, "env": env, "bound": null,
        "started_ms": started_ms});
    let record_path = sibling(&config.model, &format!(".stub-{}.json", std::process::id()));
    println!("stub: loading {}", config.alias);
    if behaviour.exit_during_load {
        let _ = std::fs::write(&record_path, record.to_string());
        std::thread::sleep(behaviour.load / 2);
        eprintln!("stub: failed to load the model");
        std::process::exit(3);
    }
    std::thread::sleep(behaviour.bind_delay);
    let listener = match TcpListener::bind((config.host.as_str(), config.port)) {
        Ok(listener) => listener,
        Err(_) => {
            let _ = std::fs::write(&record_path, record.to_string());
            std::process::exit(2)
        }
    };
    if let Ok(address) = listener.local_addr() {
        record["bound"] = json!(address.to_string());
    }
    let _ = std::fs::write(&record_path, record.to_string());
    if let Some(after) = behaviour.exit_after {
        std::thread::spawn(move || {
            std::thread::sleep(after);
            std::process::exit(4);
        });
    }
    let server = Arc::new(Server {
        log: sibling(&config.model, ".stub-requests.log"),
        config,
        behaviour,
        token,
        started: Instant::now(),
    });
    for stream in listener.incoming().flatten() {
        let server = server.clone();
        std::thread::spawn(move || serve(&server, stream));
    }
}
