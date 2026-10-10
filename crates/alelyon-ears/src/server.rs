//! The ears as a service: one loopback socket, shared by the window and by Sinai's hearing.
//!
//! Clients connect to `ws://127.0.0.1:<port>/?token=<token>`; the port and token are written to
//! `~/.alelyon/angel/ears.json` at start, readable by this user only (their home folder). A request that
//! carries an `Origin` header comes from a browser page and is refused, so no website can reach the ears.
//!
//! Commands (JSON objects with `cmd`, and an optional `id` echoed in the reply):
//!   {"cmd":"state"}
//!   {"cmd":"listen","source":"mic"|"pc","on":true|false}   the microphone for captions and Sinai; the PC's audio
//!   {"cmd":"dictate","on":true|false}                       the microphone for typing; events say purpose "dictation"
//!   {"cmd":"speaking","on":true|false}                      Sinai is talking: raise the bar for starting speech
//!   {"cmd":"transcribe_file","path":"...","formats":["txt","srt","vtt","json"],"out_dir":"..."}
//!   {"cmd":"cancel","job":"job-3"}
//! Every reply is `ears.reply` with `ok`. Events: `ears.state` (whenever anything is switched, and to each new
//! client: the on-air truth the window shows), `ears.speech`, `ears.partial`, `ears.final`, `ears.error` (from
//! `events`), and `ears.file` for file jobs (state, progress, outputs), which drives the progress bars.

use std::collections::HashMap;
use std::io::ErrorKind;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use tungstenite::handshake::server::{ErrorResponse, Request, Response};
use tungstenite::protocol::WebSocketConfig;
use tungstenite::Message;

use crate::capture::{Capture, Endpoint};
use crate::events;
use crate::files;
use crate::resample;
use crate::stream::{Config as StreamConfig, Event, Source, Transcriber, RATE};
use crate::wav;
use crate::whisper::WhisperServer;

pub struct Settings {
    pub bind: SocketAddr,
    pub recognizer: SocketAddr,
    pub language: Option<String>,
    /// Offered to the recogniser for the microphone, e.g. the spelling of "Sinai".
    pub mic_prompt: Option<String>,
    pub token: String,
    pub model: String,
}

/// Every connected client's outbox.
#[derive(Clone, Default)]
pub struct Hub {
    outboxes: Arc<Mutex<Vec<Sender<String>>>>,
}

impl Hub {
    pub fn subscribe(&self) -> Receiver<String> {
        let (tx, rx) = mpsc::channel();
        self.outboxes.lock().unwrap().push(tx);
        rx
    }

    pub fn send(&self, v: &Value) {
        let text = v.to_string();
        self.outboxes.lock().unwrap().retain(|tx| tx.send(text.clone()).is_ok());
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MicUse {
    listen: bool,
    dictation: bool,
}

/// What one client has asked for. A device is on while any connected client wants it, and a client that goes
/// away (closed, crashed, cut off) withdraws what it asked for: the microphone never stays on air for a window
/// that is no longer there.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Wants {
    listen: bool,
    dictation: bool,
    pc: bool,
}

/// Opens a device for live capture: the real one, or a stand-in in tests (no microphone is opened by a test).
type Opener = fn(Endpoint, Sender<Vec<f32>>) -> Result<Option<Capture>, String>;

fn open_device(endpoint: Endpoint, tx: Sender<Vec<f32>>) -> Result<Option<Capture>, String> {
    Capture::start(endpoint, tx).map(Some)
}

impl MicUse {
    fn any(self) -> bool {
        self.listen || self.dictation
    }

    fn purpose(self) -> &'static str {
        if self.dictation {
            "dictation"
        } else {
            "listen"
        }
    }
}

struct Live {
    capture: Option<Capture>,
    device: String,
    session: String,
}

struct Job {
    id: String,
    path: String,
    state: &'static str,
    progress: f64,
    cancel: Arc<AtomicBool>,
}

pub struct Service {
    settings: Settings,
    hub: Hub,
    live: Mutex<HashMap<&'static str, Live>>,
    mic_use: Arc<Mutex<MicUse>>,
    /// Each connected client's wishes, by client number (0 is this process's own caller).
    wants: Mutex<HashMap<u64, Wants>>,
    clients: AtomicU64,
    open: Opener,
    speaking: Arc<AtomicBool>,
    jobs: Mutex<Vec<Job>>,
    counter: AtomicU64,
    recognizer_state: Mutex<&'static str>,
}

/// Jobs kept in `ears.state` after they end, newest first.
const JOBS_KEPT: usize = 20;

impl Service {
    pub fn new(settings: Settings) -> Arc<Service> {
        Self::with_opener(settings, open_device)
    }

    fn with_opener(settings: Settings, open: Opener) -> Arc<Service> {
        Arc::new(Service {
            settings,
            hub: Hub::default(),
            live: Mutex::new(HashMap::new()),
            mic_use: Arc::new(Mutex::new(MicUse::default())),
            wants: Mutex::new(HashMap::new()),
            clients: AtomicU64::new(0),
            open,
            speaking: Arc::new(AtomicBool::new(false)),
            jobs: Mutex::new(Vec::new()),
            counter: AtomicU64::new(1),
            recognizer_state: Mutex::new("ready"),
        })
    }

    pub fn hub(&self) -> &Hub {
        &self.hub
    }

    pub fn set_recognizer_state(&self, state: &'static str) {
        *self.recognizer_state.lock().unwrap() = state;
        self.announce();
    }

    fn next_id(&self, kind: &str) -> String {
        format!("{kind}-{}", self.counter.fetch_add(1, Ordering::SeqCst))
    }

    pub fn state(&self) -> Value {
        let live = self.live.lock().unwrap();
        let mic_use = *self.mic_use.lock().unwrap();
        let mic = match live.get("mic") {
            Some(l) => json!({"on": true, "device": l.device, "session": l.session, "purpose": mic_use.purpose(),
                              "listen": mic_use.listen, "dictation": mic_use.dictation}),
            None => json!({"on": false}),
        };
        let pc = match live.get("pc") {
            Some(l) => json!({"on": true, "device": l.device, "session": l.session}),
            None => json!({"on": false}),
        };
        let jobs: Vec<Value> = self
            .jobs
            .lock()
            .unwrap()
            .iter()
            .map(|j| json!({"job": j.id, "path": j.path, "state": j.state, "progress": (j.progress * 1000.0).round() / 1000.0}))
            .collect();
        json!({
            "type": "ears.state",
            "mic": mic,
            "pc": pc,
            "recognizer": {"state": *self.recognizer_state.lock().unwrap(), "model": self.settings.model},
            "jobs": jobs,
        })
    }

    fn announce(&self) {
        self.hub.send(&self.state());
    }

    /// One command from this process's own caller; the reply.
    pub fn command(self: &Arc<Self>, cmd: &Value) -> Value {
        self.command_from(0, cmd)
    }

    /// One command from client `client`; the reply.
    fn command_from(self: &Arc<Self>, client: u64, cmd: &Value) -> Value {
        let id = cmd.get("id").cloned().unwrap_or(Value::Null);
        let name = cmd.get("cmd").and_then(Value::as_str).unwrap_or("").to_string();
        let on = cmd.get("on").and_then(Value::as_bool);
        let result: Result<Value, String> = match name.as_str() {
            "state" => Ok(self.state()),
            "listen" => match (cmd.get("source").and_then(Value::as_str), on) {
                (Some("mic"), Some(on)) => self.want(client, |w| w.listen = on),
                (Some("pc"), Some(on)) => self.want(client, |w| w.pc = on),
                _ => Err("listen needs \"source\": \"mic\" or \"pc\", and \"on\": true or false".into()),
            },
            "dictate" => match on {
                Some(on) => self.want(client, |w| w.dictation = on),
                None => Err("dictate needs \"on\": true or false".into()),
            },
            "speaking" => match on {
                Some(on) => {
                    self.speaking.store(on, Ordering::SeqCst);
                    Ok(Value::Null)
                }
                None => Err("speaking needs \"on\": true or false".into()),
            },
            "transcribe_file" => self.start_job(cmd),
            "cancel" => match cmd.get("job").and_then(Value::as_str) {
                Some(job) => self.cancel(job),
                None => Err("cancel needs \"job\"".into()),
            },
            "" => Err("a command needs \"cmd\"".into()),
            other => Err(format!("unknown command {other:?}")),
        };
        match result {
            Ok(v) => json!({"type": "ears.reply", "id": id, "cmd": name, "ok": true, "result": v}),
            Err(e) => json!({"type": "ears.reply", "id": id, "cmd": name, "ok": false, "error": e}),
        }
    }

    /// Change what `client` wants, then start or stop the devices to match everyone's wishes. A device that
    /// cannot start leaves the client's wish as it was.
    fn want(self: &Arc<Self>, client: u64, change: impl FnOnce(&mut Wants)) -> Result<Value, String> {
        let before = {
            let mut wants = self.wants.lock().unwrap();
            let entry = wants.entry(client).or_default();
            let before = *entry;
            change(entry);
            before
        };
        let result = self.follow_wants();
        if result.is_err() {
            self.wants.lock().unwrap().insert(client, before);
            let _ = self.follow_wants();
        }
        self.announce();
        result.map(|()| Value::Null)
    }

    /// Everyone's wishes together: a device is wanted while any client wants it.
    fn wanted(&self) -> Wants {
        self.wants.lock().unwrap().values().fold(Wants::default(), |all, w| Wants {
            listen: all.listen || w.listen,
            dictation: all.dictation || w.dictation,
            pc: all.pc || w.pc,
        })
    }

    /// Start or stop the microphone and the PC's audio so they match what the clients want.
    fn follow_wants(self: &Arc<Self>) -> Result<(), String> {
        let wanted = self.wanted();
        let mic = MicUse { listen: wanted.listen, dictation: wanted.dictation };
        *self.mic_use.lock().unwrap() = mic;
        let (mic_on, pc_on) = {
            let live = self.live.lock().unwrap();
            (live.contains_key("mic"), live.contains_key("pc"))
        };
        let mut result = Ok(());
        match (mic.any(), mic_on) {
            (true, false) => result = self.start_live(Source::Mic),
            (false, true) => self.stop_live("mic"),
            _ => {}
        }
        match (wanted.pc, pc_on) {
            (true, false) => {
                if let Err(e) = self.start_live(Source::Pc) {
                    result = result.and(Err(e));
                }
            }
            (false, true) => self.stop_live("pc"),
            _ => {}
        }
        result
    }

    /// Client `client` has gone: what it asked for is withdrawn, and a device nobody else wants stops.
    fn release(self: &Arc<Self>, client: u64) {
        let had = self.wants.lock().unwrap().remove(&client);
        if had.is_some_and(|w| w != Wants::default()) {
            let _ = self.follow_wants();
            self.announce();
        }
    }

    fn start_live(self: &Arc<Self>, source: Source) -> Result<(), String> {
        let (endpoint, key) = match source {
            Source::Mic => (Endpoint::Microphone, "mic"),
            _ => (Endpoint::Loopback, "pc"),
        };
        let (tx, rx) = mpsc::channel();
        let capture = (self.open)(endpoint, tx)?;
        let session = self.next_id(key);
        let (rate, channels, device) = match &capture {
            Some(c) => (c.rate(), c.channels(), c.device_name().to_string()),
            None => (RATE, 1, "a stand-in device".to_string()),
        };
        self.live.lock().unwrap().insert(key, Live { capture, device, session: session.clone() });
        let svc = Arc::clone(self);
        std::thread::Builder::new()
            .name(format!("ears-{key}"))
            .spawn(move || svc.transcribe_live(source, rx, rate, channels, session))
            .map_err(|e| format!("cannot start the {key} transcriber: {e}"))?;
        Ok(())
    }

    fn stop_live(&self, key: &str) {
        // Dropping the capture stops it; its channel closes, and the transcriber finishes the utterance in
        // progress and ends on its own thread.
        if let Some(mut live) = self.live.lock().unwrap().remove(key) {
            live.capture.take();
        }
    }

    fn transcribe_live(self: Arc<Self>, source: Source, rx: Receiver<Vec<f32>>, rate: u32, channels: usize, session: String) {
        let mut recognizer = WhisperServer::new(self.settings.recognizer);
        recognizer.language = self.settings.language.clone();
        let mut config = StreamConfig::live(source);
        if source == Source::Mic {
            config.base_prompt = self.settings.mic_prompt.clone();
        }
        let mut transcriber = Transcriber::new(recognizer, config);
        let mut resampler = resample::Resampler::new(rate, RATE);
        loop {
            match rx.recv_timeout(Duration::from_millis(250)) {
                Ok(block) => {
                    if source == Source::Mic {
                        transcriber.set_speaking(self.speaking.load(Ordering::SeqCst));
                    }
                    let mono = resample::downmix(&block, channels);
                    for e in transcriber.push(&resampler.process(&mono)) {
                        self.emit(&e, &session, source);
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        for e in transcriber.finish() {
            self.emit(&e, &session, source);
        }
    }

    fn emit(&self, e: &Event, session: &str, source: Source) {
        let mut v = events::event(e, session);
        let purpose = if source == Source::Mic { self.mic_use.lock().unwrap().purpose() } else { "pc" };
        v["purpose"] = json!(purpose);
        self.hub.send(&v);
    }

    fn start_job(self: &Arc<Self>, cmd: &Value) -> Result<Value, String> {
        let path = PathBuf::from(cmd.get("path").and_then(Value::as_str).ok_or("transcribe_file needs \"path\"")?);
        if !path.is_file() {
            return Err(format!("no such file: {}", path.display()));
        }
        let formats: Vec<String> = match cmd.get("formats").and_then(Value::as_array) {
            Some(list) => list.iter().filter_map(Value::as_str).map(str::to_string).collect(),
            None => vec!["txt".to_string()],
        };
        if formats.is_empty() || formats.iter().any(|f| !["txt", "srt", "vtt", "json"].contains(&f.as_str())) {
            return Err("formats must be some of txt, srt, vtt, json".into());
        }
        let out_dir = match cmd.get("out_dir").and_then(Value::as_str) {
            Some(d) => PathBuf::from(d),
            None => path.parent().map(Path::to_path_buf).unwrap_or_default(),
        };
        if !out_dir.is_dir() {
            return Err(format!("no such folder to save into: {}", out_dir.display()));
        }
        let id = self.next_id("job");
        let cancel = Arc::new(AtomicBool::new(false));
        {
            let mut jobs = self.jobs.lock().unwrap();
            jobs.insert(0, Job { id: id.clone(), path: path.display().to_string(), state: "decoding", progress: 0.0, cancel: cancel.clone() });
            jobs.truncate(JOBS_KEPT);
        }
        self.announce();
        let svc = Arc::clone(self);
        let job = id.clone();
        std::thread::Builder::new()
            .name(id.clone())
            .spawn(move || svc.run_job(&job, &path, &formats, &out_dir, &cancel))
            .map_err(|e| format!("cannot start the job: {e}"))?;
        Ok(json!({"job": id}))
    }

    fn job_update(&self, id: &str, state: &'static str, progress: f64) {
        if let Some(j) = self.jobs.lock().unwrap().iter_mut().find(|j| j.id == id) {
            j.state = state;
            j.progress = progress;
        }
    }

    fn run_job(&self, id: &str, path: &Path, formats: &[String], out_dir: &Path, cancel: &AtomicBool) {
        let report = |state: &'static str, extra: Value| {
            let mut v = json!({"type": "ears.file", "job": id, "path": path.display().to_string(), "state": state});
            if let (Some(obj), Value::Object(more)) = (v.as_object_mut(), extra) {
                obj.extend(more);
            }
            self.hub.send(&v);
        };
        report("decoding", json!({"progress": 0.0}));
        let audio = match load_16k(path) {
            Ok(a) => a,
            Err(e) => {
                self.job_update(id, "failed", 0.0);
                report("failed", json!({"error": e}));
                self.announce();
                return;
            }
        };
        let mut recognizer = WhisperServer::new(self.settings.recognizer);
        recognizer.language = self.settings.language.clone();
        let mut last = -1i64;
        self.job_update(id, "transcribing", 0.0);
        self.announce();
        let result = files::transcribe(&audio, recognizer, None, |f| {
            let pct = (f * 100.0).floor() as i64;
            if pct != last {
                last = pct;
                self.job_update(id, "transcribing", f);
                report("transcribing", json!({"progress": (f * 1000.0).round() / 1000.0}));
            }
            !cancel.load(Ordering::SeqCst)
        });
        if result.cancelled {
            self.job_update(id, "cancelled", 0.0);
            report("cancelled", json!({}));
            self.announce();
            return;
        }
        let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "transcript".into());
        let mut outputs = Vec::new();
        for format in formats {
            let text = match format.as_str() {
                "srt" => files::to_srt(&result),
                "vtt" => files::to_vtt(&result),
                "json" => serde_json::to_string_pretty(&files::to_json(&result, &path.display().to_string())).unwrap_or_default(),
                _ => files::to_text(&result),
            };
            let target = unique(out_dir, &stem, format);
            match std::fs::write(&target, text) {
                Ok(()) => outputs.push(target.display().to_string()),
                Err(e) => report("failed", json!({"error": format!("{}: {e}", target.display())})),
            }
        }
        self.job_update(id, "done", 1.0);
        report("done", json!({"progress": 1.0, "outputs": outputs, "lines": result.lines.len(), "errors": result.errors, "seconds": result.seconds}));
        self.announce();
    }

    fn cancel(&self, id: &str) -> Result<Value, String> {
        let jobs = self.jobs.lock().unwrap();
        let job = jobs.iter().find(|j| j.id == id).ok_or_else(|| format!("no job {id}"))?;
        job.cancel.store(true, Ordering::SeqCst);
        Ok(Value::Null)
    }

    /// Accept clients until the listener fails.
    pub fn serve(self: Arc<Self>, listener: TcpListener) -> Result<(), String> {
        for stream in listener.incoming() {
            match stream {
                Ok(s) => {
                    let svc = Arc::clone(&self);
                    let _ = std::thread::Builder::new().name("ears-client".into()).spawn(move || svc.client(s));
                }
                Err(e) => eprintln!("[ears] a connection failed: {e}"),
            }
        }
        Ok(())
    }

    fn client(self: Arc<Self>, stream: TcpStream) {
        let token = self.settings.token.clone();
        let check = move |req: &Request, resp: Response| -> Result<Response, ErrorResponse> {
            if req.headers().contains_key("origin") {
                return Err(refuse(403, "browser pages may not connect to the ears"));
            }
            let presented = req.uri().query().unwrap_or("").split('&').find_map(|kv| kv.strip_prefix("token="));
            match presented {
                Some(t) if same(t.as_bytes(), token.as_bytes()) => Ok(resp),
                _ => Err(refuse(401, "a valid token is required")),
            }
        };
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
        let mut config = WebSocketConfig::default();
        config.max_message_size = Some(1 << 20);
        config.max_frame_size = Some(1 << 20);
        let Ok(mut ws) = tungstenite::accept_hdr_with_config(stream, check, Some(config)) else {
            return;
        };
        let _ = ws.get_ref().set_read_timeout(Some(Duration::from_millis(50)));
        let me = self.clients.fetch_add(1, Ordering::SeqCst) + 1;
        // However this client's loop ends, what it asked for ends with it.
        let _leaving = Leaving { service: Arc::clone(&self), client: me };
        let outbox = self.hub.subscribe();
        if ws.send(Message::text(self.state().to_string())).is_err() {
            return;
        }
        loop {
            while let Ok(text) = outbox.try_recv() {
                if ws.send(Message::text(text)).is_err() {
                    return;
                }
            }
            match ws.read() {
                Ok(Message::Text(text)) => {
                    let reply = match serde_json::from_str::<Value>(&text) {
                        Ok(cmd) => self.command_from(me, &cmd),
                        Err(e) => json!({"type": "ears.reply", "ok": false, "error": format!("not JSON: {e}")}),
                    };
                    if ws.send(Message::text(reply.to_string())).is_err() {
                        return;
                    }
                }
                Ok(Message::Close(_)) => return,
                Ok(_) => {}
                Err(tungstenite::Error::Io(e)) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
                Err(_) => return,
            }
        }
    }
}

/// Withdraws a client's wishes when its connection's loop ends, by any path.
struct Leaving {
    service: Arc<Service>,
    client: u64,
}

impl Drop for Leaving {
    fn drop(&mut self) {
        self.service.release(self.client);
    }
}

fn refuse(status: u16, why: &str) -> ErrorResponse {
    let mut r = ErrorResponse::new(Some(why.to_string()));
    *r.status_mut() = tungstenite::http::StatusCode::from_u16(status).unwrap_or(tungstenite::http::StatusCode::FORBIDDEN);
    r
}

/// Equal, compared in time that does not depend on where they differ.
fn same(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// `out_dir/stem.ext`, or `stem (2).ext` and so on when that exists: a transcript never overwrites a file.
fn unique(out_dir: &Path, stem: &str, ext: &str) -> PathBuf {
    let first = out_dir.join(format!("{stem}.{ext}"));
    if !first.exists() {
        return first;
    }
    (2..)
        .map(|n| out_dir.join(format!("{stem} ({n}).{ext}")))
        .find(|p| !p.exists())
        .expect("an unused name exists")
}

/// Any supported file as 16 kHz mono.
pub fn load_16k(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let audio = match wav::decode(&bytes) {
        Ok(a) => a,
        Err(wav::WavError::NotWav) => crate::media::decode_file(path)?,
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let mono = resample::downmix(&audio.samples, usize::from(audio.channels));
    Ok(resample::resample_all(&mono, audio.rate, RATE))
}

/// 32 random bytes from the system's generator, as hex: the clients' token.
pub fn new_token() -> Result<String, String> {
    use windows::Win32::Security::Cryptography::{BCryptGenRandom, BCRYPT_USE_SYSTEM_PREFERRED_RNG};
    let mut bytes = [0u8; 32];
    // SAFETY: the buffer is ours and the system's preferred generator needs no handle.
    let status = unsafe { BCryptGenRandom(None, &mut bytes, BCRYPT_USE_SYSTEM_PREFERRED_RNG) };
    if status.is_err() {
        return Err(format!("the system random generator failed: {status:?}"));
    }
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// Where the clients find the port and token: `~/.alelyon/angel/ears.json`.
pub fn connection_file() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE").map(|home| PathBuf::from(home).join(".alelyon").join("angel").join("ears.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> Settings {
        Settings {
            bind: "127.0.0.1:0".parse().unwrap(),
            recognizer: "127.0.0.1:9".parse().unwrap(),
            language: None,
            mic_prompt: None,
            token: "secret".into(),
            model: "test".into(),
        }
    }

    /// Opens nothing: no test turns a real microphone on.
    fn stand_in(_: Endpoint, _: Sender<Vec<f32>>) -> Result<Option<Capture>, String> {
        Ok(None)
    }

    fn broken(_: Endpoint, _: Sender<Vec<f32>>) -> Result<Option<Capture>, String> {
        Err("no such device".into())
    }

    fn service() -> Arc<Service> {
        Service::with_opener(settings(), stand_in)
    }

    #[test]
    fn a_client_that_leaves_takes_its_microphone_with_it() {
        let svc = service();
        assert_eq!(svc.command_from(1, &json!({"cmd": "listen", "source": "mic", "on": true}))["ok"], true);
        assert_eq!(svc.command_from(2, &json!({"cmd": "dictate", "on": true}))["ok"], true);
        assert_eq!(svc.state()["mic"]["on"], true);
        svc.release(1);
        let mic = svc.state()["mic"].clone();
        assert_eq!((mic["on"].clone(), mic["listen"].clone(), mic["purpose"].clone()), (json!(true), json!(false), json!("dictation")), "client 2 still dictates: {mic}");
        svc.release(2);
        assert_eq!(svc.state()["mic"]["on"], false, "nobody wants it, so it is off air");
        assert_eq!(svc.command_from(3, &json!({"cmd": "listen", "source": "pc", "on": true}))["ok"], true);
        assert_eq!(svc.state()["pc"]["on"], true);
        svc.release(3);
        assert_eq!(svc.state()["pc"]["on"], false, "the PC's audio the same way");
        svc.release(4);
        assert_eq!(svc.wanted(), Wants::default(), "a client that asked for nothing changes nothing");
    }

    #[test]
    fn a_device_that_cannot_start_leaves_the_wish_as_it_was() {
        let svc = Service::with_opener(settings(), broken);
        let reply = svc.command_from(1, &json!({"cmd": "listen", "source": "mic", "on": true}));
        assert_eq!(reply["ok"], false);
        assert!(reply["error"].as_str().unwrap().contains("no such device"), "{reply}");
        assert_eq!(svc.state()["mic"]["on"], false);
        assert_eq!(svc.wanted(), Wants::default());
    }

    #[test]
    fn commands_are_checked_and_answered_by_id() {
        let svc = service();
        let reply = svc.command(&json!({"cmd": "listen", "source": "radio", "on": true, "id": 7}));
        assert_eq!(reply["ok"], false);
        assert_eq!(reply["id"], 7);
        assert!(reply["error"].as_str().unwrap().contains("source"));
        assert_eq!(svc.command(&json!({"cmd": "fly"}))["ok"], false);
        let state = svc.command(&json!({"cmd": "state"}));
        assert_eq!(state["ok"], true);
        assert_eq!(state["result"]["mic"]["on"], false);
        assert_eq!(state["result"]["recognizer"]["model"], "test");
    }

    #[test]
    fn a_file_job_refuses_what_it_cannot_do_before_it_starts() {
        let svc = service();
        assert!(svc.command(&json!({"cmd": "transcribe_file", "path": "C:/no/such/file.mp3"}))["error"].as_str().unwrap().contains("no such file"));
        let dir = std::env::temp_dir().join(format!("angel-ears-job-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("a.wav");
        std::fs::write(&f, crate::wav::encode_pcm16(&[0.0; 160], 16_000)).unwrap();
        let bad = svc.command(&json!({"cmd": "transcribe_file", "path": f, "formats": ["docx"]}));
        assert!(bad["error"].as_str().unwrap().contains("formats"));
        assert!(svc.jobs.lock().unwrap().is_empty(), "nothing was started");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_transcript_never_overwrites_a_file() {
        let dir = std::env::temp_dir().join(format!("angel-ears-unique-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(unique(&dir, "talk", "txt"), dir.join("talk.txt"));
        std::fs::write(dir.join("talk.txt"), "x").unwrap();
        std::fs::write(dir.join("talk (2).txt"), "x").unwrap();
        assert_eq!(unique(&dir, "talk", "txt"), dir.join("talk (3).txt"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tokens_are_random_and_compared_whole() {
        let (a, b) = (new_token().unwrap(), new_token().unwrap());
        assert_eq!(a.len(), 64);
        assert_ne!(a, b);
        assert!(same(b"abc", b"abc"));
        assert!(!same(b"abc", b"abd"));
        assert!(!same(b"abc", b"abcd"));
    }

    fn connect(addr: SocketAddr, path: &str, origin: Option<&str>) -> Result<String, String> {
        use std::io::{Read, Write};
        let mut s = TcpStream::connect(addr).map_err(|e| e.to_string())?;
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut req = format!(
            "GET {path} HTTP/1.1\r\nHost: {addr}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n"
        );
        if let Some(o) = origin {
            req.push_str(&format!("Origin: {o}\r\n"));
        }
        req.push_str("\r\n");
        s.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
        let mut buf = [0u8; 512];
        let n = s.read(&mut buf).map_err(|e| e.to_string())?;
        Ok(String::from_utf8_lossy(&buf[..n]).lines().next().unwrap_or("").to_string())
    }

    #[test]
    fn only_a_client_with_the_token_and_no_browser_origin_gets_in() {
        let svc = service();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || svc.serve(listener));
        assert!(connect(addr, "/?token=secret", None).unwrap().contains("101"), "the right token upgrades");
        assert!(connect(addr, "/?token=wrong", None).unwrap().contains("401"));
        assert!(connect(addr, "/", None).unwrap().contains("401"));
        assert!(connect(addr, "/?token=secret", Some("https://example.com")).unwrap().contains("403"), "a browser page is refused");
    }
}
