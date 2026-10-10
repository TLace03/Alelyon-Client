//! The recogniser: whisper.cpp's own `whisper-server`, spoken to over loopback HTTP.
//!
//! Any build of whisper.cpp's server (`setup` says where `serve` finds one), asked for `verbose_json`,
//! which carries segments with times and, inside them, words with times and probabilities (read from the
//! bundled build on 2026-10-03: segment keys id, text, start, end, tokens, words, temperature, avg_logprob,
//! no_speech_prob; word keys word, start, end, t_dtw, probability). Every call is bounded: a connect
//! timeout, then one deadline for writing the request and reading the whole answer, so a stalled server
//! costs at most `timeout` and never a hung thread.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::wav;

/// One word with its place in the audio, in seconds from the start of the audio that was sent.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    /// As the recogniser wrote it, leading space included, so concatenating words rebuilds the text.
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub probability: f32,
}

impl Word {
    /// Lower case, letters and digits only: how two hypotheses are compared.
    pub fn key(&self) -> String {
        normalize(&self.text)
    }

    pub fn shifted(&self, by: f64) -> Word {
        Word { start: self.start + by, end: self.end + by, ..self.clone() }
    }
}

pub fn normalize(text: &str) -> String {
    text.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect()
}

/// Words back to text: concatenated, then trimmed.
pub fn join(words: &[Word]) -> String {
    let mut out = String::new();
    for w in words {
        out.push_str(&w.text);
    }
    out.trim().to_string()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub words: Vec<Word>,
    pub no_speech_prob: f32,
    pub avg_logprob: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transcript {
    pub language: Option<String>,
    pub segments: Vec<Segment>,
}

impl Transcript {
    pub fn words(&self) -> Vec<Word> {
        self.segments.iter().flat_map(|s| s.words.iter().cloned()).collect()
    }

    pub fn text(&self) -> String {
        self.segments.iter().map(|s| s.text.trim()).filter(|t| !t.is_empty()).collect::<Vec<_>>().join(" ")
    }
}

/// Whisper's own rule for a segment heard in silence: likely no speech AND a low average log-probability.
/// Such segments are where it writes "Thank you." over an empty room, so they are dropped.
pub const NO_SPEECH_PROB: f32 = 0.6;
pub const LOW_LOGPROB: f32 = -1.0;

fn hallucinated(seg: &Segment) -> bool {
    seg.no_speech_prob > NO_SPEECH_PROB && seg.avg_logprob < LOW_LOGPROB
}

/// Anything that turns 16 kHz mono audio into words. The ears hold one per source; tests hold a fake.
pub trait Recognizer: Send {
    fn transcribe(&mut self, audio: &[f32], prompt: Option<&str>) -> Result<Transcript, String>;
}

/// `whisper-server`'s `/inference`.
#[derive(Clone, Debug)]
pub struct WhisperServer {
    pub addr: SocketAddr,
    pub path: String,
    /// None for the server's own default; "auto" to detect.
    pub language: Option<String>,
    pub timeout: Duration,
    pub connect_timeout: Duration,
}

impl WhisperServer {
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            path: "/inference".to_string(),
            language: Some("auto".to_string()),
            timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(2),
        }
    }

    /// Is a server answering at all? A bounded `GET /`, as the loop's own `whisper_check` does.
    pub fn alive(&self) -> bool {
        let Ok(mut stream) = TcpStream::connect_timeout(&self.addr, self.connect_timeout) else {
            return false;
        };
        let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
        let request = format!("GET / HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n", self.addr);
        if stream.write_all(request.as_bytes()).is_err() {
            return false;
        }
        let mut head = [0u8; 12];
        matches!(stream.read(&mut head), Ok(n) if n >= 12 && &head[..5] == b"HTTP/")
    }
}

impl Recognizer for WhisperServer {
    fn transcribe(&mut self, audio: &[f32], prompt: Option<&str>) -> Result<Transcript, String> {
        let wav = wav::encode_pcm16(audio, 16_000);
        let mut fields: Vec<(&str, String)> =
            vec![("response_format", "verbose_json".into()), ("temperature", "0.0".into())];
        if let Some(lang) = &self.language {
            fields.push(("language", lang.clone()));
        }
        if let Some(p) = prompt.filter(|p| !p.trim().is_empty()) {
            fields.push(("prompt", p.to_string()));
        }
        let (status, body) = post_multipart(self, &fields, &wav)?;
        if status != 200 {
            let snippet = String::from_utf8_lossy(&body[..body.len().min(200)]).to_string();
            return Err(format!("whisper-server answered {status}: {snippet}"));
        }
        parse_verbose_json(&body)
    }
}

/// `verbose_json` into a transcript, segments heard in silence dropped.
pub fn parse_verbose_json(body: &[u8]) -> Result<Transcript, String> {
    let v: Value = serde_json::from_slice(body).map_err(|e| format!("whisper-server sent JSON that does not parse: {e}"))?;
    if let Some(err) = v.get("error").and_then(Value::as_str) {
        return Err(format!("whisper-server: {err}"));
    }
    let language = v.get("language").and_then(Value::as_str).map(str::to_string);
    let mut segments = Vec::new();
    for s in v.get("segments").and_then(Value::as_array).into_iter().flatten() {
        let num = |k: &str| s.get(k).and_then(Value::as_f64).unwrap_or(0.0);
        let words = s
            .get("words")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|w| {
                let text = w.get("word")?.as_str()?.to_string();
                Some(Word {
                    text,
                    start: w.get("start").and_then(Value::as_f64).unwrap_or(0.0),
                    end: w.get("end").and_then(Value::as_f64).unwrap_or(0.0),
                    probability: w.get("probability").and_then(Value::as_f64).unwrap_or(0.0) as f32,
                })
            })
            .filter(|w| !w.text.is_empty())
            .collect();
        let words = merge_pieces(words);
        let seg = Segment {
            start: num("start"),
            end: num("end"),
            text: s.get("text").and_then(Value::as_str).unwrap_or("").to_string(),
            words,
            no_speech_prob: num("no_speech_prob") as f32,
            avg_logprob: num("avg_logprob") as f32,
        };
        if !hallucinated(&seg) {
            segments.push(seg);
        }
    }
    Ok(Transcript { language, segments: mend(segments) })
}

/// Whisper can end a segment inside a word: " ... cake, donuts and t" then "arts." (read from the bundled
/// server, 2026-10-03, LibriSpeech 4992-41806-0011). Pieces at the start of a segment that continue the word
/// before are moved back onto it, and each segment's text is rebuilt from its words, so no line ends half a
/// word. A segment left with no words is dropped.
fn mend(segments: Vec<Segment>) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::with_capacity(segments.len());
    for mut seg in segments {
        let had_words = !seg.words.is_empty();
        while let (Some(prev), Some(first)) = (out.last_mut(), seg.words.first()) {
            let continues = !first.text.starts_with(char::is_whitespace) || first.key().is_empty();
            let Some(last) = prev.words.last_mut().filter(|_| continues) else { break };
            let piece = seg.words.remove(0);
            last.text.push_str(&piece.text);
            last.end = last.end.max(piece.end);
            last.probability = last.probability.min(piece.probability);
            prev.end = prev.end.max(piece.end);
        }
        if had_words && seg.words.is_empty() {
            continue;
        }
        if let Some(first) = seg.words.first() {
            seg.start = seg.start.max(first.start.min(seg.end));
        }
        out.push(seg);
    }
    for seg in &mut out {
        if !seg.words.is_empty() {
            seg.text = seg.words.iter().map(|w| w.text.as_str()).collect();
        }
    }
    out
}

/// Whisper writes some words as pieces: "o'clock" as " o", "'", "clock", "seven-year" as " seven", "-",
/// "year". A piece with no leading space, or with no letter or digit in it, belongs to the word before it,
/// so it is joined to that word (its end time too). Without this the apostrophe and the hyphen, which have
/// nothing to compare, were lost and the words ran together ("oclock"): measured 2026-10-03 on LibriSpeech.
pub fn merge_pieces(words: Vec<Word>) -> Vec<Word> {
    let mut out: Vec<Word> = Vec::with_capacity(words.len());
    for w in words {
        let continues = !w.text.starts_with(char::is_whitespace) || w.key().is_empty();
        match out.last_mut() {
            Some(prev) if continues => {
                prev.text.push_str(&w.text);
                prev.end = prev.end.max(w.end);
                prev.probability = prev.probability.min(w.probability);
            }
            _ => out.push(w),
        }
    }
    out
}

fn post_multipart(server: &WhisperServer, fields: &[(&str, String)], wav: &[u8]) -> Result<(u16, Vec<u8>), String> {
    let deadline = Instant::now() + server.timeout;
    let boundary = format!("----angel-ears-{:x}", Instant::now().elapsed().as_nanos() ^ std::process::id() as u128);
    let mut body = Vec::with_capacity(wav.len() + 1024);
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes());
    }
    body.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(wav);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let head = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: multipart/form-data; boundary={boundary}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        server.path,
        server.addr,
        body.len()
    );

    let mut stream = TcpStream::connect_timeout(&server.addr, server.connect_timeout)
        .map_err(|e| format!("cannot reach whisper-server at {}: {e}", server.addr))?;
    let remaining = |what: &str| {
        deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or_else(|| format!("whisper-server did not finish {what} within {:?}", server.timeout))
    };
    stream.set_write_timeout(Some(remaining("the request")?)).map_err(|e| e.to_string())?;
    stream.write_all(head.as_bytes()).and_then(|_| stream.write_all(&body)).map_err(|e| format!("sending to whisper-server: {e}"))?;

    let mut response = Vec::new();
    let mut chunk = [0u8; 16 * 1024];
    loop {
        stream.set_read_timeout(Some(remaining("its answer")?)).map_err(|e| e.to_string())?;
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => response.extend_from_slice(&chunk[..n]),
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {
                return Err(format!("whisper-server did not finish its answer within {:?}", server.timeout));
            }
            Err(e) => return Err(format!("reading whisper-server's answer: {e}")),
        }
    }
    parse_http_response(&response)
}

/// Status and body of a complete HTTP/1.1 response (Content-Length, chunked, or read to close).
pub fn parse_http_response(raw: &[u8]) -> Result<(u16, Vec<u8>), String> {
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| "whisper-server's answer has no header end".to_string())?;
    let head = String::from_utf8_lossy(&raw[..split]);
    let rest = &raw[split + 4..];
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("bad status line from whisper-server: {status_line:?}"))?;
    let mut length = None;
    let mut chunked = false;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
            if k == "content-length" {
                length = v.parse::<usize>().ok();
            } else if k == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked") {
                chunked = true;
            }
        }
    }
    let body = if chunked {
        dechunk(rest)?
    } else if let Some(n) = length {
        if rest.len() < n {
            return Err(format!("whisper-server's answer stopped at {} of {n} bytes", rest.len()));
        }
        rest[..n].to_vec()
    } else {
        rest.to_vec()
    };
    Ok((status, body))
}

fn dechunk(mut data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let line_end = data.windows(2).position(|w| w == b"\r\n").ok_or("chunked answer ends early")?;
        let size_text = String::from_utf8_lossy(&data[..line_end]);
        let size = usize::from_str_radix(size_text.split(';').next().unwrap_or("").trim(), 16)
            .map_err(|_| format!("bad chunk size {size_text:?}"))?;
        data = &data[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        if data.len() < size + 2 {
            return Err("chunked answer ends inside a chunk".into());
        }
        out.extend_from_slice(&data[..size]);
        data = &data[size + 2..];
    }
}

/// A recogniser that answers from a script: each call takes the next transcript. For tests.
pub struct Scripted {
    pub answers: std::collections::VecDeque<Result<Transcript, String>>,
    pub calls: Vec<(usize, Option<String>)>,
}

impl Scripted {
    pub fn new(answers: Vec<Result<Transcript, String>>) -> Self {
        Self { answers: answers.into(), calls: Vec::new() }
    }
}

impl Recognizer for Scripted {
    fn transcribe(&mut self, audio: &[f32], prompt: Option<&str>) -> Result<Transcript, String> {
        self.calls.push((audio.len(), prompt.map(str::to_string)));
        self.answers.pop_front().unwrap_or_else(|| Ok(Transcript::default()))
    }
}

/// Words, spaced, at even times from `start`, `step` seconds apart: a transcript for tests.
pub fn transcript_of(text: &str, start: f64, step: f64) -> Transcript {
    let words: Vec<Word> = text
        .split_whitespace()
        .enumerate()
        .map(|(i, w)| Word {
            text: format!(" {w}"),
            start: start + i as f64 * step,
            end: start + (i as f64 + 0.8) * step,
            probability: 0.9,
        })
        .collect();
    let end = words.last().map(|w| w.end).unwrap_or(start);
    Transcript {
        language: Some("en".into()),
        segments: vec![Segment { start, end, text: format!(" {text}"), words, no_speech_prob: 0.01, avg_logprob: -0.1 }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    const VERBOSE: &str = r#"{"task":"transcribe","language":"english","duration":3.2,"text":" He hoped",
      "segments":[{"id":0,"text":" He hoped there","start":0.0,"end":3.28,"tokens":[1,2],
        "words":[{"word":" He","start":0.0,"end":0.15,"t_dtw":-1,"probability":0.67},
                 {"word":" hoped","start":0.53,"end":0.53,"t_dtw":-1,"probability":0.99},
                 {"word":" there","start":0.57,"end":0.84,"t_dtw":-1,"probability":0.99}],
        "temperature":0.0,"avg_logprob":-0.07,"no_speech_prob":0.04},
       {"id":1,"text":" Thank you.","start":3.3,"end":4.0,"tokens":[3],
        "words":[{"word":" Thank","start":3.3,"end":3.6,"probability":0.2},{"word":" you.","start":3.6,"end":4.0,"probability":0.2}],
        "temperature":0.0,"avg_logprob":-1.4,"no_speech_prob":0.92}]}"#;

    #[test]
    fn verbose_json_reads_and_silence_hallucinations_are_dropped() {
        let t = parse_verbose_json(VERBOSE.as_bytes()).unwrap();
        assert_eq!(t.language.as_deref(), Some("english"));
        assert_eq!(t.segments.len(), 1, "the 'Thank you.' heard in silence is dropped");
        let words = t.words();
        assert_eq!(join(&words), "He hoped there");
        assert_eq!(words[1].start, 0.53);
        assert_eq!(t.text(), "He hoped there");
    }

    #[test]
    fn pieces_of_a_word_are_joined_and_punctuation_kept() {
        let w = |t: &str, s: f64| Word { text: t.into(), start: s, end: s + 0.1, probability: 0.9 };
        let merged = merge_pieces(vec![w(" by", 0.0), w(" 9", 0.2), w(" o", 0.4), w("'", 0.5), w("clock", 0.6), w(" seven", 0.8), w("-", 0.9), w("year", 1.0), w(" slave.", 1.2)]);
        assert_eq!(join(&merged), "by 9 o'clock seven-year slave.");
        assert_eq!(merged.iter().map(|w| w.text.trim()).collect::<Vec<_>>(), ["by", "9", "o'clock", "seven-year", "slave."]);
        assert!((merged[2].end - 0.7).abs() < 1e-9, "the joined word ends where its last piece ends");
    }

    #[test]
    fn a_word_split_between_segments_is_mended() {
        let body = r#"{"language":"en","segments":[
          {"text":" cake, donuts and t","start":4.0,"end":7.6,"avg_logprob":-0.2,"no_speech_prob":0.01,
           "words":[{"word":" cake","start":4.0,"end":4.4,"probability":0.9},{"word":",","start":4.4,"end":4.4,"probability":0.9},
                    {"word":" don","start":4.5,"end":4.7,"probability":0.9},{"word":"uts","start":4.7,"end":5.0,"probability":0.9},
                    {"word":" and","start":5.1,"end":5.3,"probability":0.9},{"word":" t","start":5.4,"end":7.6,"probability":0.5}]},
          {"text":"arts.","start":7.6,"end":8.2,"avg_logprob":-0.3,"no_speech_prob":0.02,
           "words":[{"word":"arts","start":7.6,"end":8.0,"probability":0.8},{"word":".","start":8.0,"end":8.2,"probability":0.9}]}]}"#;
        let t = parse_verbose_json(body.as_bytes()).unwrap();
        assert_eq!(t.segments.len(), 1, "the segment holding only the rest of a word is folded back");
        assert_eq!(t.segments[0].text.trim(), "cake, donuts and tarts.");
        assert_eq!(t.text(), "cake, donuts and tarts.");
        assert_eq!(t.segments[0].end, 8.2);
    }

    #[test]
    fn a_server_error_is_an_error() {
        assert!(parse_verbose_json(br#"{"error":"failed to read audio"}"#).unwrap_err().contains("failed to read audio"));
        assert!(parse_verbose_json(b"not json").is_err());
    }

    #[test]
    fn http_bodies_by_length_chunks_and_close() {
        let (s, b) = parse_http_response(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhelloEXTRA").unwrap();
        assert_eq!((s, b.as_slice()), (200, &b"hello"[..]));
        let (s, b) = parse_http_response(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2;x=y\r\nde\r\n0\r\n\r\n").unwrap();
        assert_eq!((s, b.as_slice()), (200, &b"abcde"[..]));
        let (s, b) = parse_http_response(b"HTTP/1.1 500 Internal\r\nConnection: close\r\n\r\noops").unwrap();
        assert_eq!((s, b.as_slice()), (500, &b"oops"[..]));
        assert!(parse_http_response(b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\nshort").is_err());
    }

    #[test]
    fn normalize_ignores_case_and_punctuation() {
        assert_eq!(normalize(" Dinner,"), "dinner");
        assert_eq!(normalize("Don't!"), "dont");
    }

    /// An HTTP 200 carrying `body`, its length counted rather than written by hand.
    fn ok_json(body: &str) -> &'static [u8] {
        let answer = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len());
        Box::leak(answer.into_bytes().into_boxed_slice())
    }

    /// A one-shot loopback server: answers the first request with `answer` after `delay`.
    fn serve_once(answer: &'static [u8], delay: Duration) -> (SocketAddr, std::thread::JoinHandle<Vec<u8>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            sock.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut got = Vec::new();
            let mut buf = [0u8; 65536];
            // read until the closing boundary has arrived
            while !got.windows(4).any(|w| w == b"--\r\n") {
                match sock.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => got.extend_from_slice(&buf[..n]),
                }
            }
            std::thread::sleep(delay);
            let _ = sock.write_all(answer);
            got
        });
        (addr, handle)
    }

    #[test]
    fn a_request_reaches_the_server_and_its_answer_comes_back() {
        let answer = ok_json(r#"{"language":"en","segments":[{"text":" Hi","start":0,"end":1,"words":[{"word":" Hi","start":0.1,"end":0.4,"probability":0.9}]}]}"#);
        let (addr, server) = serve_once(answer, Duration::ZERO);
        let mut rec = WhisperServer::new(addr);
        let t = rec.transcribe(&[0.0; 1600], Some("Sinai")).unwrap();
        assert_eq!(t.text(), "Hi");
        let request = String::from_utf8_lossy(&server.join().unwrap()).to_string();
        assert!(request.starts_with("POST /inference HTTP/1.1"));
        assert!(request.contains("name=\"response_format\"\r\n\r\nverbose_json"));
        assert!(request.contains("name=\"prompt\"\r\n\r\nSinai"));
        assert!(request.contains("filename=\"audio.wav\""));
    }

    #[test]
    fn a_server_that_stalls_costs_at_most_the_timeout() {
        static LATE: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}";
        let (addr, _server) = serve_once(LATE, Duration::from_secs(3));
        let mut rec = WhisperServer::new(addr);
        rec.timeout = Duration::from_millis(400);
        let began = Instant::now();
        let err = rec.transcribe(&[0.0; 1600], None).unwrap_err();
        let took = began.elapsed();
        assert!(err.contains("did not finish"), "{err}");
        assert!(took < Duration::from_millis(1500), "{took:?}");
    }

    #[test]
    fn an_absent_server_is_reported_not_waited_for() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let mut rec = WhisperServer::new(addr);
        assert!(!rec.alive());
        assert!(rec.transcribe(&[0.0; 160], None).unwrap_err().contains("cannot reach"));
    }
}
