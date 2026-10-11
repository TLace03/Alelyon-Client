//! The ears in the cloud: speech to text on the person's own provider account, in place of whisper-server.
//!
//! `angel-ears serve --cloud groq [--cloud-model whisper-large-v3-turbo]` reads with the provider's
//! OpenAI-compatible `POST <base>/audio/transcriptions` (multipart: the utterance as a 16 kHz WAV, the model,
//! `response_format=verbose_json`, word and segment times, the language unless it is `auto`, the prompt). The
//! key is the person's own, read from the environment under the provider's key name (`GROQ_API_KEY`); the
//! window that starts the ears hands it over only for a part the person agreed to send. It is never logged,
//! shown or put in an error.
//!
//! Live speech is read once per utterance, when it ends (no readings while it is spoken), so a provider's
//! per-minute limit is not spent on settling captions. When the provider says its free limit is reached (a
//! `429`), the ears pause until the time it gives ([`Pause`]): every reading until then fails at once with
//! "free limit reached, back at <time>", nothing is sent, and the service's state says `paused` with
//! `back_at`. Nothing retries, switches provider or pays.
//!
//! The transport is the system's own HTTP stack (WinHTTP, `winhttp`), so the ears link no TLS library. A debug
//! build honours `ALELYON_CLOUD_STUB` (an `http://` address on this machine) in place of the provider's origin,
//! for tests and screenshots.

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::wav;
use crate::whisper::{merge_pieces, Recognizer, Segment, Transcript, Word, LOW_LOGPROB, NO_SPEECH_PROB};

/// A provider the ears can read with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Provider {
    pub id: &'static str,
    pub label: &'static str,
    /// The OpenAI-compatible base address.
    pub base: &'static str,
    /// Where its key is read from in the environment.
    pub key_env: &'static str,
    /// Its speech-to-text model when none is named.
    pub model: &'static str,
}

/// The providers, as their own pages give their addresses (read 2026-10-10).
pub const PROVIDERS: &[Provider] = &[Provider {
    id: "groq",
    label: "Groq",
    base: "https://api.groq.com/openai/v1",
    key_env: "GROQ_API_KEY",
    model: "whisper-large-v3-turbo",
}];

pub fn provider(id: &str) -> Option<&'static Provider> {
    PROVIDERS.iter().find(|p| p.id == id)
}

/// The environment switch a debug build reads for a stub provider on this machine.
pub const STUB_ENV: &str = "ALELYON_CLOUD_STUB";

/// A key that prints as `***`.
#[derive(Clone)]
pub struct Key(String);

impl Key {
    pub fn new(key: impl Into<String>) -> Key {
        Key(key.into().trim().to_string())
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("***")
    }
}

/// `origin` when it is an `http://` address on this machine with nothing after the port.
pub fn loopback_origin(origin: &str) -> Option<String> {
    let rest = origin.trim().strip_prefix("http://")?.trim_end_matches('/');
    let (host, port) = rest.rsplit_once(':')?;
    let loopback = matches!(host, "127.0.0.1" | "localhost" | "[::1]");
    (loopback && !port.is_empty() && port.chars().all(|c| c.is_ascii_digit())).then(|| format!("http://{host}:{port}"))
}

/// What the ears read with in the cloud.
#[derive(Clone, Debug)]
pub struct Spec {
    pub provider: &'static Provider,
    /// The base address, the provider's own or a stub's.
    pub base: String,
    pub model: String,
    pub key: Key,
    /// None or "auto": the model tells.
    pub language: Option<String>,
    pub timeout: Duration,
}

impl Spec {
    /// `provider` with `key`, at its own address or at `stub`'s origin.
    pub fn new(provider: &'static Provider, model: Option<&str>, key: Key, stub: Option<&str>) -> Result<Spec, String> {
        if key.expose().is_empty() {
            return Err(format!("no {} key: set {} for the ears to read in the cloud", provider.label, provider.key_env));
        }
        let base = match stub {
            Some(s) => {
                let origin = loopback_origin(s).ok_or_else(|| format!("{STUB_ENV} must be an http:// address on this machine"))?;
                let path = provider.base.splitn(4, '/').nth(3).unwrap_or("");
                format!("{origin}/{path}")
            }
            None => provider.base.to_string(),
        };
        let model = model.map(str::trim).filter(|m| !m.is_empty()).unwrap_or(provider.model).to_string();
        Ok(Spec { provider, base, model, key, language: Some("auto".into()), timeout: Duration::from_secs(60) })
    }

    /// The model's name as the service reports it.
    pub fn name(&self) -> String {
        format!("{} ({}, in the cloud)", self.model, self.provider.label)
    }
}

/// A free limit reached: when the provider said to come back.
#[derive(Debug, Default)]
pub struct Pause {
    until: Mutex<Option<i64>>,
}

impl Pause {
    /// The time the pause ends, while it lasts at `now`.
    pub fn until(&self, now: i64) -> Option<i64> {
        let mut until = self.until.lock().unwrap();
        match *until {
            Some(t) if t > now => Some(t),
            Some(_) => {
                *until = None;
                None
            }
            None => None,
        }
    }

    pub fn set(&self, at: i64) {
        let mut until = self.until.lock().unwrap();
        *until = Some(until.map_or(at, |t| t.max(at)));
    }
}

/// Seconds since 1970, now.
pub fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// "13:05Z": a time of day in UTC, as the service says it (the window shows it in local time).
pub fn clock_utc(at: i64) -> String {
    let secs = at.rem_euclid(86_400);
    format!("{:02}:{:02}Z", secs / 3600, secs % 3600 / 60)
}

/// What a request came back with.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reply {
    pub status: u16,
    /// Lower-case names.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Reply {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.trim())
    }
}

/// The seam to the provider: the system's HTTP stack (`winhttp`), or a test's.
pub trait Transport: Send + Sync {
    /// POST `body` to `url` with these headers, the whole answer within `timeout`.
    fn post(&self, url: &str, headers: &[(String, String)], body: &[u8], timeout: Duration) -> Result<Reply, String>;
}

/// A duration as Groq writes it (`2m59.56s`, `7.66s`, `120ms`), in seconds.
pub fn duration_text(text: &str) -> Option<f64> {
    let text = text.trim();
    if let Ok(n) = text.parse::<f64>() {
        return (n.is_finite() && n >= 0.0).then_some(n);
    }
    let (mut total, mut number) = (0.0, String::new());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() || c == '.' {
            number.push(c);
            continue;
        }
        let n: f64 = number.parse().ok()?;
        number.clear();
        total += n * match (c, chars.peek()) {
            ('m', Some('s')) => {
                chars.next();
                0.001
            }
            ('h', _) => 3600.0,
            ('m', _) => 60.0,
            ('s', _) => 1.0,
            _ => return None,
        };
    }
    (number.is_empty() && !text.is_empty()).then_some(total)
}

/// When to come back after a `429`: `Retry-After` in seconds, else the provider's request reset, else a
/// minute. Never before `now`.
pub fn back_at(reply: &Reply, now: i64) -> i64 {
    let after = reply.header("retry-after").and_then(|v| v.parse::<f64>().ok()).filter(|n| n.is_finite() && *n >= 0.0);
    let reset = || ["x-ratelimit-reset-requests", "x-ratelimit-reset"].iter().find_map(|n| reply.header(n)).and_then(duration_text);
    let wait = after.or_else(reset).unwrap_or(60.0);
    now + wait.ceil().max(0.0) as i64
}

/// A multipart form: the fields, then the audio as `file`.
pub fn form(fields: &[(&str, String)], wav: &[u8], boundary: &str) -> Vec<u8> {
    let mut body = Vec::with_capacity(wav.len() + 1024);
    for (name, value) in fields {
        let value = value.replace(['\r', '\n'], " ");
        body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes());
    }
    body.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(wav);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

/// OpenAI's `verbose_json` (top-level `segments` and `words`, words without their leading space) as the ears'
/// transcript: each word goes to the segment its start falls in, with the leading space the ears' words carry;
/// a segment without words is cut into words over its time. Segments heard in silence are dropped, by
/// Whisper's own rule.
pub fn parse_verbose(body: &[u8]) -> Result<Transcript, String> {
    let v: Value = serde_json::from_slice(body).map_err(|_| "the provider's answer could not be read".to_string())?;
    let num = |o: &Value, k: &str| o.get(k).and_then(Value::as_f64).unwrap_or(0.0);
    let words: Vec<Word> = v
        .get("words")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|w| {
            let text = w.get("word")?.as_str()?.trim();
            (!text.is_empty()).then(|| Word { text: format!(" {text}"), start: num(w, "start"), end: num(w, "end"), probability: 1.0 })
        })
        .collect();
    let mut segments: Vec<Segment> = v
        .get("segments")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|s| Segment {
            start: num(s, "start"),
            end: num(s, "end"),
            text: s.get("text").and_then(Value::as_str).unwrap_or("").to_string(),
            words: Vec::new(),
            no_speech_prob: num(s, "no_speech_prob") as f32,
            avg_logprob: num(s, "avg_logprob") as f32,
        })
        .collect();
    if segments.is_empty() {
        let text = v.get("text").and_then(Value::as_str).ok_or("the provider's answer held no text")?;
        let end = words.last().map_or(0.0, |w| w.end);
        segments.push(Segment { start: 0.0, end, text: text.to_string(), words: Vec::new(), no_speech_prob: 0.0, avg_logprob: 0.0 });
    }
    let count = segments.len();
    for w in words {
        let at = segments.iter().position(|s| w.start < s.end).unwrap_or(count - 1);
        segments[at].words.push(w);
    }
    for s in &mut segments {
        if s.words.is_empty() {
            let pieces: Vec<&str> = s.text.split_whitespace().collect();
            let step = (s.end - s.start).max(0.0) / pieces.len().max(1) as f64;
            s.words = pieces
                .iter()
                .enumerate()
                .map(|(i, p)| Word { text: format!(" {p}"), start: s.start + step * i as f64, end: s.start + step * (i + 1) as f64, probability: 1.0 })
                .collect();
        }
        s.words = merge_pieces(std::mem::take(&mut s.words));
        s.text = s.words.iter().map(|w| w.text.as_str()).collect();
    }
    segments.retain(|s| !(s.no_speech_prob > NO_SPEECH_PROB && s.avg_logprob < LOW_LOGPROB) && !s.words.is_empty());
    Ok(Transcript { language: v.get("language").and_then(Value::as_str).map(str::to_string), segments })
}

/// Speech to text in the cloud.
pub struct CloudRecognizer {
    spec: Arc<Spec>,
    transport: Arc<dyn Transport>,
    pause: Arc<Pause>,
    clock: fn() -> i64,
}

impl CloudRecognizer {
    pub fn new(spec: Arc<Spec>, transport: Arc<dyn Transport>, pause: Arc<Pause>) -> CloudRecognizer {
        CloudRecognizer { spec, transport, pause, clock: now }
    }

    /// With a clock of a test's own.
    pub fn with_clock(mut self, clock: fn() -> i64) -> CloudRecognizer {
        self.clock = clock;
        self
    }
}

/// What a paused reading says.
pub fn paused_words(until: i64) -> String {
    format!("free limit reached, back at {}", clock_utc(until))
}

impl Recognizer for CloudRecognizer {
    fn transcribe(&mut self, audio: &[f32], prompt: Option<&str>) -> Result<Transcript, String> {
        let now = (self.clock)();
        if let Some(until) = self.pause.until(now) {
            return Err(paused_words(until));
        }
        let spec = &self.spec;
        let wav = wav::encode_pcm16(audio, 16_000);
        let mut fields: Vec<(&str, String)> = vec![
            ("model", spec.model.clone()),
            ("response_format", "verbose_json".into()),
            ("temperature", "0".into()),
            ("timestamp_granularities[]", "word".into()),
            ("timestamp_granularities[]", "segment".into()),
        ];
        if let Some(lang) = spec.language.as_deref().filter(|l| !l.is_empty() && *l != "auto") {
            fields.push(("language", lang.to_string()));
        }
        if let Some(p) = prompt.filter(|p| !p.trim().is_empty()) {
            fields.push(("prompt", p.to_string()));
        }
        let boundary = format!("angel-ears-{:x}{:x}", now, std::process::id());
        let body = form(&fields, &wav, &boundary);
        let headers = vec![
            ("Authorization".to_string(), format!("Bearer {}", spec.key.expose())),
            ("Content-Type".to_string(), format!("multipart/form-data; boundary={boundary}")),
        ];
        let label = spec.provider.label;
        let reply = self.transport.post(&format!("{}/audio/transcriptions", spec.base), &headers, &body, spec.timeout)?;
        match reply.status {
            200..=299 => parse_verbose(&reply.body),
            429 => {
                let until = back_at(&reply, now);
                self.pause.set(until);
                Err(paused_words(until))
            }
            401 | 403 => Err(format!("{label} did not accept the key: make a new one and connect again")),
            402 => Err(format!("{label} asks for payment: the ears use free models only, so nothing was paid")),
            status => Err(format!("{label} answered with an error ({status})")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request as sent: its address, headers and body.
    type Sent = (String, Vec<(String, String)>, Vec<u8>);

    /// Answers every request with one reply and keeps what it was sent.
    struct Canned {
        reply: Reply,
        seen: Mutex<Vec<Sent>>,
    }

    impl Transport for Canned {
        fn post(&self, url: &str, headers: &[(String, String)], body: &[u8], _: Duration) -> Result<Reply, String> {
            self.seen.lock().unwrap().push((url.to_string(), headers.to_vec(), body.to_vec()));
            Ok(self.reply.clone())
        }
    }

    const NOW: i64 = 1_791_633_600;

    fn spec() -> Arc<Spec> {
        Arc::new(Spec::new(provider("groq").unwrap(), None, Key::new("stub-key-for-tests"), None).unwrap())
    }

    fn canned(status: u16, headers: Vec<(&str, &str)>, body: &str) -> Arc<Canned> {
        Arc::new(Canned {
            reply: Reply {
                status,
                headers: headers.into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                body: body.as_bytes().to_vec(),
            },
            seen: Mutex::new(Vec::new()),
        })
    }

    #[test]
    fn a_spec_is_the_providers_address_or_a_stub_on_this_machine() {
        let s = spec();
        assert_eq!(s.base, "https://api.groq.com/openai/v1");
        assert_eq!(s.model, "whisper-large-v3-turbo");
        let stubbed = Spec::new(provider("groq").unwrap(), Some("whisper-large-v3"), Key::new("k"), Some("http://127.0.0.1:4567")).unwrap();
        assert_eq!(stubbed.base, "http://127.0.0.1:4567/openai/v1");
        assert_eq!(stubbed.model, "whisper-large-v3");
        for bad in ["https://127.0.0.1:1", "http://example.com:80", "http://127.0.0.1:1/x"] {
            assert!(Spec::new(provider("groq").unwrap(), None, Key::new("k"), Some(bad)).is_err(), "{bad}");
        }
        assert!(Spec::new(provider("groq").unwrap(), None, Key::new(" "), None).is_err(), "no key, no cloud");
        assert_eq!(format!("{:?}", Key::new("secret")), "***");
    }

    #[test]
    fn an_utterance_goes_as_a_wav_form_and_its_words_come_back_with_their_times() {
        let t = canned(
            200,
            vec![],
            r#"{"text":" Hey Sinai, it's seven.","language":"english",
                "segments":[{"start":0.0,"end":1.0,"text":" Hey Sinai,","avg_logprob":-0.1,"no_speech_prob":0.0},
                            {"start":1.0,"end":2.0,"text":" it's seven.","avg_logprob":-0.1,"no_speech_prob":0.0}],
                "words":[{"word":"Hey","start":0.0,"end":0.3},{"word":"Sinai,","start":0.3,"end":0.9},
                         {"word":"it's","start":1.1,"end":1.4},{"word":"seven.","start":1.4,"end":1.9}]}"#,
        );
        let mut r = CloudRecognizer::new(spec(), t.clone(), Arc::new(Pause::default())).with_clock(|| NOW);
        let heard = r.transcribe(&vec![0.0; 16_000], Some("Hey Sinai")).unwrap();
        assert_eq!(heard.text(), "Hey Sinai, it's seven.");
        let words = heard.words();
        assert_eq!(words.len(), 4);
        assert_eq!(words[2].text, " it's");
        assert_eq!((words[3].start, words[3].end), (1.4, 1.9));
        let seen = t.seen.lock().unwrap();
        let (url, headers, body) = &seen[0];
        assert_eq!(url, "https://api.groq.com/openai/v1/audio/transcriptions");
        assert!(headers.iter().any(|(k, v)| k == "Authorization" && v == "Bearer stub-key-for-tests"));
        let body = String::from_utf8_lossy(body);
        assert!(body.contains("name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n"));
        assert!(body.contains("name=\"prompt\"\r\n\r\nHey Sinai\r\n"));
        assert!(body.contains("filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF"));
        assert!(!body.contains("name=\"language\""), "auto: the model tells");
    }

    #[test]
    fn a_limit_pauses_the_ears_until_the_time_given_and_nothing_more_is_sent() {
        let t = canned(429, vec![("retry-after", "90")], "{}");
        let pause = Arc::new(Pause::default());
        let mut r = CloudRecognizer::new(spec(), t.clone(), pause.clone()).with_clock(|| NOW);
        let said = r.transcribe(&vec![0.0; 8_000], None).unwrap_err();
        assert_eq!(said, format!("free limit reached, back at {}", clock_utc(NOW + 90)));
        assert_eq!(pause.until(NOW), Some(NOW + 90));
        let again = r.transcribe(&vec![0.0; 8_000], None).unwrap_err();
        assert!(again.starts_with("free limit reached"));
        assert_eq!(t.seen.lock().unwrap().len(), 1, "a paused reading sends nothing");
        assert_eq!(pause.until(NOW + 90), None, "the pause ends at its time");
        assert_eq!(back_at(&Reply { status: 429, headers: vec![("x-ratelimit-reset-requests".into(), "2m30s".into())], body: vec![] }, NOW), NOW + 150);
        assert_eq!(back_at(&Reply::default(), NOW), NOW + 60);
    }

    #[test]
    fn a_providers_error_is_said_without_its_text_or_the_key() {
        for (status, words) in [(401, "did not accept the key"), (402, "nothing was paid"), (500, "error (500)")] {
            let t = canned(status, vec![], "{\"error\":\"detail stub-key-for-tests\"}");
            let mut r = CloudRecognizer::new(spec(), t, Arc::new(Pause::default())).with_clock(|| NOW);
            let said = r.transcribe(&vec![0.0; 8_000], None).unwrap_err();
            assert!(said.contains(words), "{said}");
            assert!(!said.contains("stub-key") && !said.contains("detail"), "{said}");
        }
    }

    #[test]
    fn words_without_times_are_spread_over_their_segment_and_silence_is_dropped() {
        let t = parse_verbose(
            br#"{"text":"one two","segments":[{"start":0,"end":2,"text":" one two","avg_logprob":-0.2,"no_speech_prob":0.1},
                {"start":2,"end":3,"text":" Thank you.","avg_logprob":-1.5,"no_speech_prob":0.9}]}"#,
        )
        .unwrap();
        assert_eq!(t.segments.len(), 1, "the segment heard in silence is dropped");
        let w = t.words();
        assert_eq!((w[0].text.as_str(), w[0].start, w[0].end), (" one", 0.0, 1.0));
        assert_eq!((w[1].text.as_str(), w[1].start, w[1].end), (" two", 1.0, 2.0));
        let plain = parse_verbose(br#"{"text":"just text"}"#).unwrap();
        assert_eq!(plain.text(), "just text");
        assert!(parse_verbose(b"not json").is_err());
        assert_eq!(duration_text("7.66s"), Some(7.66));
        assert_eq!(clock_utc(NOW + 3 * 3600 + 5 * 60), "15:05Z");
    }
}
