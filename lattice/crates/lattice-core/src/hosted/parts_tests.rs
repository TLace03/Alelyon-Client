//! The parts' routes against a stub provider on this machine: what each route
//! sends, what it reads back, a provider's error, a limit with the time to try
//! again, and a model that is not free refused before anything is sent.
//! Nothing here reaches a provider, a real key or Credential Manager.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::*;

/// A request as the stub saw it.
#[derive(Clone, Debug, Default)]
struct Seen {
    method: String,
    path: String,
    auth: Option<String>,
    content_type: Option<String>,
    body: Vec<u8>,
}

impl Seen {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}

/// A reply the stub gives: status, extra headers, body.
type Reply = (u16, Vec<(&'static str, String)>, Vec<u8>);

/// A provider on this machine that answers each request with the next reply
/// (the last one again once they run out) and records what it was sent.
struct Stub {
    addr: SocketAddr,
    seen: Arc<Mutex<Vec<Seen>>>,
}

fn read_request(stream: &mut std::net::TcpStream) -> Seen {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        let n = stream.read(&mut chunk).unwrap_or(0);
        if n == 0 {
            break buf.len();
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end.min(buf.len())]).to_string();
    let mut lines = head.lines();
    let mut first = lines.next().unwrap_or("").split(' ');
    let (method, path) = (first.next().unwrap_or("").to_owned(), first.next().unwrap_or("").to_owned());
    let mut seen = Seen { method, path, ..Seen::default() };
    let mut length = 0usize;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            match k.trim().to_ascii_lowercase().as_str() {
                "authorization" => seen.auth = Some(v.trim().to_owned()),
                "content-type" => seen.content_type = Some(v.trim().to_owned()),
                "content-length" => length = v.trim().parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    let mut body = buf[head_end.min(buf.len())..].to_vec();
    while body.len() < length {
        let n = stream.read(&mut chunk).unwrap_or(0);
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    seen.body = body;
    seen
}

impl Stub {
    fn new(replies: Vec<Reply>) -> Stub {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            let mut replies = replies.into_iter();
            let mut last: Option<Reply> = None;
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let request = read_request(&mut stream);
                log.lock().unwrap().push(request);
                let (status, headers, body) = replies.next().or_else(|| last.clone()).unwrap();
                last = Some((status, headers.clone(), body.clone()));
                let mut head = format!("HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n", body.len());
                if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-type")) {
                    head.push_str("Content-Type: application/json\r\n");
                }
                for (k, v) in &headers {
                    head.push_str(&format!("{k}: {v}\r\n"));
                }
                head.push_str("\r\n");
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body);
            }
        });
        Stub { addr, seen }
    }

    fn json(status: u16, body: Value) -> Stub {
        Stub::new(vec![(status, Vec::new(), body.to_string().into_bytes())])
    }

    fn origin(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

/// A key made up for the stub; never a real one.
const KEY: &str = "stub-key-for-tests";
/// A Cloudflare account id made up for the stub.
const ACCOUNT: &str = "0123456789abcdef0123456789abcdef";
/// The clock the tests run at (2026-10-10 12:00:00 UTC).
const NOW: i64 = 1_791_633_600;

/// A meter in a folder of the test's own (kept for the test's life).
fn meter() -> Arc<Meter> {
    let dir: &'static crate::testkit::TempDir = Box::leak(Box::new(crate::testkit::TempDir::new("parts-meter")));
    Arc::new(Meter::at(super::meter::path_in(dir.path())))
}

fn reach(cloud: &'static Cloud, stub: &Stub) -> Reach {
    let r = Reach::new(cloud, SecretString::new(KEY), Some(ACCOUNT), Some(&stub.origin())).unwrap();
    if cloud.id == CLOUDFLARE.id { r.with_meter(meter()) } else { r }
}

fn run<T>(work: impl std::future::Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(work)
}

#[test]
fn a_reach_fills_the_account_and_a_stub_stands_in_only_on_this_machine() {
    let r = Reach::new(&CLOUDFLARE, SecretString::new(KEY), Some(ACCOUNT), None).unwrap();
    assert_eq!(r.base(), format!("https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/v1"));
    assert!(!r.is_stub());
    let stubbed = Reach::new(&CLOUDFLARE, SecretString::new(KEY), Some(ACCOUNT), Some("http://127.0.0.1:9")).unwrap();
    assert_eq!(stubbed.base(), format!("http://127.0.0.1:9/client/v4/accounts/{ACCOUNT}/ai/v1"));
    assert!(stubbed.is_stub());
    for bad in ["https://127.0.0.1:9", "http://example.com", "http://127.0.0.1:9/path", "http://user@127.0.0.1:9"] {
        assert!(Reach::new(&GROQ, SecretString::new(KEY), None, Some(bad)).is_err(), "{bad} is not a stub on this machine");
    }
    for bad in [None, Some(""), Some("not an id"), Some("ab/cd")] {
        assert!(Reach::new(&CLOUDFLARE, SecretString::new(KEY), bad, None).is_err(), "{bad:?} is not an account id");
    }
    let no_key = Reach::new(&GROQ, SecretString::new("  "), None, None).unwrap_err();
    assert!(no_key.contains("Connect Groq first"), "{no_key}");
    assert!(!format!("{r:?}").contains(KEY), "a reach never prints its key");
}

#[test]
fn a_reach_is_found_with_the_key_store_and_needs_the_account_where_one_is_asked() {
    let keys = |name: &str| (name == "GROQ_API_KEY" || name == "CLOUDFLARE_API_TOKEN").then(|| SecretString::new(KEY));
    assert!(connected(&GROQ, &keys));
    assert!(!connected(&CLOUDFLARE, &keys), "a token without its account id is not connected");
    assert!(!connected(&OPENROUTER, &keys));
    assert!(Reach::found(&GROQ, &keys).is_ok());
    assert!(Reach::found(&CLOUDFLARE, &keys).is_err());
}

#[test]
fn only_free_models_are_called_and_a_paid_one_is_refused_before_anything_is_sent() {
    assert!(free_model(&OPENROUTER, "meta-llama/llama-3.3-70b-instruct:free"));
    assert!(!free_model(&OPENROUTER, "meta-llama/llama-3.3-70b-instruct"));
    assert!(!free_model(&GROQ, " "));
    let stub = Stub::json(200, json!({"choices": [{"message": {"content": "never"}}]}));
    let r = reach(&OPENROUTER, &stub);
    let c = Client::new(&r).unwrap();
    let refused = run(chat(&c, &r, "openai/gpt-4o", vec![json!({"role": "user", "content": "hi"})], 64, NOW)).unwrap_err();
    assert!(matches!(&refused, Failure::Failed(s) if s.contains("free models only")), "{refused:?}");
    assert!(stub.seen().is_empty(), "nothing was sent for a model that is not free");
}

#[test]
fn chat_sends_the_messages_with_the_key_and_reads_the_answer() {
    let stub = Stub::new(vec![(
        200,
        vec![("x-ratelimit-remaining-requests", "1999".into()), ("x-ratelimit-limit-requests", "2000".into()), ("x-ratelimit-reset-requests", "43.2s".into())],
        json!({"choices": [{"message": {"role": "assistant", "content": "Hello."}}]}).to_string().into_bytes(),
    )]);
    let r = reach(&GROQ, &stub);
    let c = Client::new(&r).unwrap();
    let answer = run(chat(&c, &r, "llama-3.3-70b-versatile", vec![json!({"role": "user", "content": "hi"})], 64, NOW)).unwrap();
    assert_eq!(answer.value, "Hello.");
    assert_eq!(answer.limits, Limits { remaining: Some(1999), limit: Some(2000), resets_at: Some(NOW + 44) });
    let seen = &stub.seen()[0];
    assert_eq!((seen.method.as_str(), seen.path.as_str()), ("POST", "/openai/v1/chat/completions"));
    assert_eq!(seen.auth.as_deref(), Some(format!("Bearer {KEY}").as_str()));
    let body = seen.json();
    assert_eq!(body["model"], "llama-3.3-70b-versatile");
    assert_eq!(body["stream"], false);
    assert_eq!(body["messages"][0]["content"], "hi");
}

#[test]
fn a_limit_is_a_pause_until_the_time_the_provider_gives() {
    // Retry-After in seconds
    let stub = Stub::new(vec![(429, vec![("retry-after", "120".into())], b"{}".to_vec())]);
    let r = reach(&GROQ, &stub);
    let c = Client::new(&r).unwrap();
    let got = run(chat(&c, &r, "llama-3.3-70b-versatile", vec![json!({"role": "user", "content": "hi"})], 64, NOW)).unwrap_err();
    assert_eq!(got, Failure::Limited { back_at: NOW + 120 });
    assert_eq!(stub.seen().len(), 1, "one request, no retry");

    // OpenRouter's reset in milliseconds since 1970, no Retry-After
    let reset_ms = (NOW + 3600) * 1000;
    let stub = Stub::new(vec![(429, vec![("x-ratelimit-reset", reset_ms.to_string())], b"{}".to_vec())]);
    let r = reach(&OPENROUTER, &stub);
    let c = Client::new(&r).unwrap();
    let got = run(chat(&c, &r, "qwen/qwen3-8b:free", vec![], 8, NOW)).unwrap_err();
    assert_eq!(got, Failure::Limited { back_at: NOW + 3600 });

    // Cloudflare's daily allocation spent (its error 4006): back at the next 00:00 UTC
    let stub = Stub::json(400, json!({"success": false, "errors": [{"code": 4006, "message": "daily free allocation used"}]}));
    let r = reach(&CLOUDFLARE, &stub);
    let c = Client::new(&r).unwrap();
    let got = run(embed(&c, &r, "@cf/baai/bge-m3", &["a".into()], NOW)).unwrap_err();
    assert_eq!(got, Failure::Limited { back_at: next_utc_midnight(NOW) });
    assert_eq!(next_utc_midnight(NOW), NOW + 12 * 3600);
}

#[test]
fn retry_times_are_read_in_each_providers_way() {
    let got = |headers: Vec<(&str, &str)>| Got {
        status: 429,
        headers: headers.into_iter().map(|(k, v)| (k.to_owned(), v.to_owned())).collect(),
        body: Vec::new(),
    };
    assert_eq!(back_at(&GROQ, &got(vec![("retry-after", "2")]), NOW), NOW + 2);
    assert_eq!(back_at(&GROQ, &got(vec![("x-ratelimit-reset-requests", "2m59.56s")]), NOW), NOW + 180);
    assert_eq!(back_at(&GROQ, &got(vec![("retry-after", "Sat, 10 Oct 2026 13:00:00 GMT")]), NOW), NOW + 3600);
    assert_eq!(back_at(&GROQ, &got(vec![]), NOW), NOW + 60, "no time given: a minute");
    assert_eq!(back_at(&CLOUDFLARE, &got(vec![]), NOW), next_utc_midnight(NOW));
    assert_eq!(back_at(&GROQ, &got(vec![("retry-after", "Sat, 10 Oct 2020 13:00:00 GMT")]), NOW), NOW, "never in the past");
    assert_eq!(duration_text("1h2m3s"), Some(3723.0));
    assert_eq!(duration_text("120ms"), Some(0.12));
    assert_eq!(duration_text("soon"), None);
}

#[test]
fn a_providers_error_is_one_sentence_without_its_text_or_the_key() {
    for (status, words) in [(401, "did not accept the key"), (402, "nothing was paid"), (404, "does not offer"), (500, "error (500)")] {
        let stub = Stub::new(vec![(status, Vec::new(), format!("{{\"error\": \"secret detail {KEY}\"}}").into_bytes())]);
        let r = reach(&GROQ, &stub);
        let c = Client::new(&r).unwrap();
        let failure = run(embed(&c, &r, "m", &["a".into()], NOW)).unwrap_err();
        let sentence = failure.sentence();
        assert!(sentence.contains(words), "{status}: {sentence}");
        assert!(!sentence.contains(KEY) && !sentence.contains("secret detail"), "{sentence}");
        assert!(!sentence.contains("127.0.0.1"), "the address is not in the sentence: {sentence}");
    }
}

#[test]
fn transcribe_sends_the_audio_as_a_multipart_form_and_reads_words_and_segments() {
    let stub = Stub::json(
        200,
        json!({
            "text": " Hey Sinai, what time is it?",
            "language": "english",
            "segments": [{"id": 0, "start": 0.0, "end": 1.8, "text": " Hey Sinai, what time is it?", "avg_logprob": -0.2, "no_speech_prob": 0.01}],
            "words": [{"word": "Hey", "start": 0.0, "end": 0.3}, {"word": "Sinai,", "start": 0.3, "end": 0.7}],
        }),
    );
    let r = reach(&GROQ, &stub);
    let c = Client::new(&r).unwrap();
    let wav = b"RIFF....WAVEfmt fake audio".to_vec();
    let heard = run(transcribe(&c, &r, "whisper-large-v3-turbo", &wav, "audio.wav", "audio/wav", Some("auto"), Some("Sinai"), NOW))
        .unwrap()
        .value;
    assert_eq!(heard.text, "Hey Sinai, what time is it?");
    assert_eq!(heard.words.len(), 2);
    assert_eq!(heard.words[1], HeardWord { word: "Sinai,".into(), start: 0.3, end: 0.7 });
    assert_eq!(heard.segments[0].end, 1.8);
    let seen = &stub.seen()[0];
    assert_eq!(seen.path, "/openai/v1/audio/transcriptions");
    let kind = seen.content_type.clone().unwrap();
    assert!(kind.starts_with("multipart/form-data; boundary="), "{kind}");
    let body = String::from_utf8_lossy(&seen.body);
    assert!(body.contains("name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n"));
    assert!(body.contains("name=\"response_format\"\r\n\r\nverbose_json\r\n"));
    assert!(body.contains("name=\"timestamp_granularities[]\"\r\n\r\nword\r\n"));
    assert!(body.contains("name=\"prompt\"\r\n\r\nSinai\r\n"));
    assert!(!body.contains("name=\"language\""), "auto lets the model tell");
    assert!(body.contains("name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF....WAVEfmt fake audio\r\n"));
}

#[test]
fn speak_reads_cloudflares_base64_envelope_raw_audio_and_openais_route() {
    let mp3 = b"ID3fake-mp3".to_vec();
    let stub = Stub::json(200, json!({"success": true, "result": {"audio": base64(&mp3)}}));
    let r = reach(&CLOUDFLARE, &stub);
    let c = Client::new(&r).unwrap();
    let speech = run(speak(&c, &r, "@cf/myshell-ai/melotts", "Hello.", None, None, NOW)).unwrap().value;
    assert_eq!(speech, Speech { bytes: mp3.clone(), mime: "audio/mpeg".into() });
    let seen = &stub.seen()[0];
    assert_eq!(seen.path, format!("/client/v4/accounts/{ACCOUNT}/ai/run/@cf/myshell-ai/melotts"));
    assert_eq!(seen.json(), json!({"prompt": "Hello.", "lang": "en"}));

    let stub = Stub::new(vec![(200, vec![("content-type", "audio/mpeg".into())], mp3.clone())]);
    let r = reach(&CLOUDFLARE, &stub);
    let c = Client::new(&r).unwrap();
    assert_eq!(run(speak(&c, &r, "@cf/myshell-ai/melotts", "Hi", None, Some("fr"), NOW)).unwrap().value.bytes, mp3);
    assert_eq!(stub.seen()[0].json()["lang"], "fr");

    let stub = Stub::new(vec![(200, vec![("content-type", "audio/wav".into())], b"RIFFwav".to_vec())]);
    let r = reach(&GROQ, &stub);
    let c = Client::new(&r).unwrap();
    let speech = run(speak(&c, &r, "playai-tts", "Hi", Some("Fritz-PlayAI"), None, NOW)).unwrap().value;
    assert_eq!(speech.mime, "audio/wav");
    let seen = &stub.seen()[0];
    assert_eq!(seen.path, "/openai/v1/audio/speech");
    assert_eq!(seen.json(), json!({"model": "playai-tts", "input": "Hi", "voice": "Fritz-PlayAI", "response_format": "wav"}));
}

#[test]
fn embed_keeps_the_inputs_order_and_refuses_a_short_answer() {
    let stub = Stub::json(200, json!({"data": [{"index": 1, "embedding": [0.5, 0.25]}, {"index": 0, "embedding": [1.0, 0.0]}]}));
    let r = reach(&CLOUDFLARE, &stub);
    let c = Client::new(&r).unwrap();
    let vectors = run(embed(&c, &r, "@cf/baai/bge-m3", &["first".into(), "second".into()], NOW)).unwrap().value;
    assert_eq!(vectors, vec![vec![1.0, 0.0], vec![0.5, 0.25]]);
    let seen = &stub.seen()[0];
    assert_eq!(seen.path, format!("/client/v4/accounts/{ACCOUNT}/ai/v1/embeddings"));
    assert_eq!(seen.json(), json!({"model": "@cf/baai/bge-m3", "input": ["first", "second"]}));

    let stub = Stub::json(200, json!({"data": [{"index": 0, "embedding": [1.0]}]}));
    let r = reach(&CLOUDFLARE, &stub);
    let c = Client::new(&r).unwrap();
    assert!(run(embed(&c, &r, "@cf/baai/bge-m3", &["a".into(), "b".into()], NOW)).is_err(), "one vector for two inputs");
}

#[test]
fn look_sends_an_image_as_a_data_url_or_as_cloudflares_bytes() {
    let png = b"\x89PNGfake".to_vec();
    let stub = Stub::json(200, json!({"choices": [{"message": {"content": [{"type": "text", "text": "A window."}]}}]}));
    let r = reach(&OPENROUTER, &stub);
    let c = Client::new(&r).unwrap();
    let said = run(look(&c, &r, "qwen/qwen2.5-vl-72b-instruct:free", "What is this?", &png, "image/png", 128, NOW)).unwrap().value;
    assert_eq!(said, "A window.");
    let body = stub.seen()[0].json();
    let parts = &body["messages"][0]["content"];
    assert_eq!(parts[0]["text"], "What is this?");
    assert_eq!(parts[1]["image_url"]["url"], format!("data:image/png;base64,{}", base64(&png)));

    let stub = Stub::json(200, json!({"success": true, "result": {"response": "A desk."}}));
    let r = reach(&CLOUDFLARE, &stub);
    let c = Client::new(&r).unwrap();
    let said = run(look(&c, &r, "@cf/meta/llama-3.2-11b-vision-instruct", "What is this?", &png, "image/png", 128, NOW)).unwrap().value;
    assert_eq!(said, "A desk.");
    let seen = &stub.seen()[0];
    assert_eq!(seen.path, format!("/client/v4/accounts/{ACCOUNT}/ai/run/@cf/meta/llama-3.2-11b-vision-instruct"));
    let body = seen.json();
    assert_eq!(body["image"], json!(png));
    assert_eq!(body["messages"][0]["content"], "What is this?");
    assert!(run(look(&c, &r, "@cf/meta/llama-3.2-11b-vision-instruct", "x", b"not", "text/plain", 8, NOW)).is_err());
}

#[test]
fn live_lists_keep_each_providers_free_models_for_the_part() {
    let stub = Stub::json(
        200,
        json!({"data": [
            {"id": "qwen/qwen3-8b:free", "name": "Qwen3 8B (free)", "hugging_face_id": "Qwen/Qwen3-8B", "pricing": {"prompt": "0", "completion": "0"},
             "architecture": {"input_modalities": ["text"]}},
            {"id": "qwen/qwen2.5-vl-72b-instruct:free", "name": "Qwen2.5 VL (free)", "hugging_face_id": "Qwen/Qwen2.5-VL-72B-Instruct",
             "pricing": {"prompt": "0", "completion": "0"}, "architecture": {"input_modalities": ["text", "image"]}},
            {"id": "qwen/qwen3-8b", "name": "Qwen3 8B", "hugging_face_id": "Qwen/Qwen3-8B", "pricing": {"prompt": "0.0000001", "completion": "0.0000002"}},
            {"id": "google/gemini-2.5-flash:free", "name": "closed", "hugging_face_id": "", "pricing": {"prompt": "0", "completion": "0"}},
        ]}),
    );
    let r = reach(&OPENROUTER, &stub);
    let c = Client::new(&r).unwrap();
    let chat_ids: Vec<String> = run(models(&c, &r, Task::Chat, NOW)).unwrap().into_iter().map(|m| m.id).collect();
    assert_eq!(chat_ids, ["qwen/qwen2.5-vl-72b-instruct:free", "qwen/qwen3-8b:free"], "free, open-weight only");
    let vision: Vec<String> = run(models(&c, &r, Task::Vision, NOW)).unwrap().into_iter().map(|m| m.id).collect();
    assert_eq!(vision, ["qwen/qwen2.5-vl-72b-instruct:free"]);

    let stub = Stub::json(200, json!({"data": [{"id": "whisper-large-v3-turbo"}, {"id": "whisper-large-v3"}, {"id": "llama-3.3-70b-versatile"}]}));
    let r = reach(&GROQ, &stub);
    let c = Client::new(&r).unwrap();
    let ears: Vec<String> = run(models(&c, &r, Task::Transcribe, NOW)).unwrap().into_iter().map(|m| m.id).collect();
    assert_eq!(ears, ["whisper-large-v3", "whisper-large-v3-turbo"]);
    assert_eq!(stub.seen()[0].auth.as_deref(), Some(format!("Bearer {KEY}").as_str()));

    let stub = Stub::json(
        200,
        json!({"success": true, "result": [
            {"id": "model-one", "name": "@cf/baai/bge-m3", "task": {"name": "Text Embeddings"}},
            {"id": "model-two", "name": "@cf/qwen/qwen3-embedding-0.6b", "task": {"name": "Text Embeddings"}},
        ]}),
    );
    let r = reach(&CLOUDFLARE, &stub);
    let c = Client::new(&r).unwrap();
    let found: Vec<String> = run(models(&c, &r, Task::Embed, NOW)).unwrap().into_iter().map(|m| m.id).collect();
    assert_eq!(found, ["@cf/baai/bge-m3", "@cf/qwen/qwen3-embedding-0.6b"]);
    let seen = &stub.seen()[0];
    assert_eq!(seen.path, format!("/client/v4/accounts/{ACCOUNT}/ai/models/search?task=Text+Embeddings&per_page=100"));
}

#[test]
fn a_licence_is_accepted_with_cloudflares_agree_prompt() {
    let stub = Stub::json(200, json!({"success": true, "result": {"response": "Thank you for agreeing."}}));
    let r = reach(&CLOUDFLARE, &stub);
    let c = Client::new(&r).unwrap();
    run(accept_licence(&c, &r, "@cf/meta/llama-3.2-11b-vision-instruct", NOW)).unwrap();
    assert_eq!(stub.seen()[0].json(), json!({"prompt": "agree"}));
}

#[test]
fn cloudflare_is_called_only_under_its_meter_and_stops_before_the_free_allocation() {
    // no meter: nothing is sent
    let stub = Stub::json(200, json!({"data": [{"index": 0, "embedding": [1.0]}]}));
    let bare = Reach::new(&CLOUDFLARE, SecretString::new(KEY), Some(ACCOUNT), Some(&stub.origin())).unwrap();
    let c = Client::new(&bare).unwrap();
    let refused = run(embed(&c, &bare, "@cf/baai/bge-m3", &["a".into()], NOW)).unwrap_err();
    assert!(refused.sentence().contains("meter"), "{refused:?}");
    // a model without a known rate: nothing is sent
    let r = reach(&CLOUDFLARE, &stub);
    let refused = run(embed(&c, &r, "@cf/unknown/embedder", &["a".into()], NOW)).unwrap_err();
    assert!(refused.sentence().contains("no published rate"), "{refused:?}");
    assert!(stub.seen().is_empty());
    // a call is counted after it answers: the provider's own token count where it gives one
    let stub = Stub::json(200, json!({"data": [{"index": 0, "embedding": [1.0]}], "usage": {"prompt_tokens": 2_000_000}}));
    let r = reach(&CLOUDFLARE, &stub);
    let c = Client::new(&r).unwrap();
    run(embed(&c, &r, "@cf/baai/bge-m3", &["hello".into()], NOW)).unwrap();
    assert!((r.meter().unwrap().used(ACCOUNT, NOW) - 2_150.0).abs() < 1e-9, "2M tokens of bge-m3 are 2,150 Neurons");
    // the voice and the eyes share the account's count; near the stop a call is refused before it is sent
    r.meter().unwrap().add(ACCOUNT, 6_849.0, NOW).unwrap();
    let voice = Reach::new(&CLOUDFLARE, SecretString::new(KEY), Some(ACCOUNT), Some(&stub.origin()))
        .unwrap()
        .with_meter(Arc::new(Meter::at(r.meter().unwrap().path().to_path_buf())));
    let before = stub.seen().len();
    let paused = run(speak(&c, &voice, "@cf/myshell-ai/melotts", &"word ".repeat(400), None, None, NOW)).unwrap_err();
    assert_eq!(paused, Failure::Limited { back_at: next_utc_midnight(NOW) }, "200 seconds of speech would pass 9,000");
    assert_eq!(stub.seen().len(), before, "nothing was sent");
}

#[test]
fn a_cloudflare_list_leaves_out_a_model_the_meter_has_no_rate_for() {
    let stub = Stub::json(
        200,
        json!({"success": true, "result": [
            {"id": "model-one", "name": "@cf/baai/bge-m3", "task": {"name": "Text Embeddings"}},
            {"id": "model-two", "name": "@cf/someone/new-embedder", "task": {"name": "Text Embeddings"}},
        ]}),
    );
    let r = reach(&CLOUDFLARE, &stub);
    let c = Client::new(&r).unwrap();
    let ids: Vec<String> = run(models(&c, &r, Task::Embed, NOW)).unwrap().into_iter().map(|m| m.id).collect();
    assert_eq!(ids, ["@cf/baai/bge-m3"]);
    // the eyes: a vision model is known by its name, not by Cloudflare's opaque id
    let stub = Stub::json(
        200,
        json!({"success": true, "result": [
            {"id": "model-three", "name": "@cf/meta/llama-3.2-11b-vision-instruct", "task": {"name": "Text Generation"}},
            {"id": "model-four", "name": "@cf/meta/llama-3.1-8b-instruct", "task": {"name": "Text Generation"}},
        ]}),
    );
    let r = reach(&CLOUDFLARE, &stub);
    let c = Client::new(&r).unwrap();
    let ids: Vec<String> = run(models(&c, &r, Task::Vision, NOW)).unwrap().into_iter().map(|m| m.id).collect();
    assert_eq!(ids, ["@cf/meta/llama-3.2-11b-vision-instruct"]);
}

#[test]
fn base64_goes_there_and_back() {
    for bytes in [&b""[..], b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar", &[0, 255, 254, 1]] {
        assert_eq!(unbase64(&base64(bytes)).unwrap(), bytes);
    }
    assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    assert_eq!(base64(b"fo"), "Zm8=");
    assert!(unbase64("not*base64").is_none());
}

#[test]
fn the_providers_are_the_four_and_only_gemini_trains_on_what_it_is_sent() {
    let ids: Vec<&str> = CLOUDS.iter().map(|c| c.id).collect();
    assert_eq!(ids, ["openrouter", "groq", "cloudflare", "gemini"]);
    assert_eq!(CLOUDS.iter().filter(|c| c.trains).map(|c| c.id).collect::<Vec<_>>(), ["gemini"]);
    for c in CLOUDS {
        assert!(c.base.starts_with("https://") && c.list.starts_with("https://") && c.key_page.starts_with("https://"), "{}", c.id);
        assert!(crate::keys::valid_key_name(c.key_name), "{}", c.key_name);
    }
}

// ------------------------------------------------------------------- paid models under a monthly cap

/// A spending ledger in a folder of the test's own (kept for the test's life).
fn ledger() -> Arc<Spend> {
    let dir: &'static crate::testkit::TempDir = Box::leak(Box::new(crate::testkit::TempDir::new("parts-spend")));
    Arc::new(Spend::at(super::spend::path_in(dir.path())))
}

fn paid(model: &str, cap: f64, ledger: &Arc<Spend>) -> Paid {
    Paid { model: model.into(), part: "mind".into(), price: Price::known(0.2, 0.6).unwrap(), cap_usd: cap, ledger: ledger.clone() }
}

#[test]
fn a_paid_model_is_refused_without_a_cap_for_that_very_model() {
    let stub = Stub::json(200, json!({"choices": [{"message": {"content": "never"}}]}));
    let l = ledger();
    // an allowance for another model, and one with no cap, admit nothing
    for allowance in [paid("qwen/qwen3-14b", 5.0, &l), paid("qwen/qwen3-235b-a22b", 0.0, &l)] {
        let r = reach(&OPENROUTER, &stub).with_paid(allowance);
        let c = Client::new(&r).unwrap();
        let refused = run(chat(&c, &r, "qwen/qwen3-235b-a22b", vec![json!({"role": "user", "content": "hi"})], 64, NOW)).unwrap_err();
        assert!(matches!(&refused, Failure::Failed(s) if s.contains("free models only")), "{refused:?}");
    }
    // speech stays free only, whatever the reach carries
    let r = reach(&GROQ, &stub).with_paid(paid("whisper-paid", 5.0, &l));
    let c = Client::new(&r).unwrap();
    assert!(run(speak(&c, &r, " ", "hello", None, None, NOW)).is_err());
    assert!(stub.seen().is_empty(), "nothing was sent");
    assert_eq!(l.used(NOW), 0.0, "nothing was counted");
}

#[test]
fn a_paid_call_is_counted_at_the_cost_the_provider_reports() {
    let stub = Stub::json(
        200,
        json!({"choices": [{"message": {"content": "Hello."}}], "usage": {"prompt_tokens": 20, "completion_tokens": 5, "cost": 0.0042}}),
    );
    let l = ledger();
    let r = reach(&OPENROUTER, &stub).with_paid(paid("qwen/qwen3-235b-a22b", 5.0, &l));
    let c = Client::new(&r).unwrap();
    let answer = run(chat(&c, &r, "qwen/qwen3-235b-a22b", vec![json!({"role": "user", "content": "hi"})], 64, NOW)).unwrap();
    assert_eq!(answer.value, "Hello.");
    assert!((l.used_by("mind", NOW) - 0.0042).abs() < 1e-12, "settled to the reported cost: {}", l.used(NOW));
    let body = stub.seen()[0].json();
    assert_eq!(body["usage"]["include"], true, "OpenRouter is asked to report the cost");
    // without a reported cost: the reported tokens at the published price
    let stub = Stub::json(200, json!({"choices": [{"message": {"content": "Hi."}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 1000}}));
    let l = ledger();
    let r = reach(&OPENROUTER, &stub).with_paid(paid("qwen/qwen3-235b-a22b", 5.0, &l));
    let c = Client::new(&r).unwrap();
    run(chat(&c, &r, "qwen/qwen3-235b-a22b", vec![json!({"role": "user", "content": "hi"})], 64, NOW)).unwrap();
    assert!((l.used(NOW) - 0.0008).abs() < 1e-12, "{}", l.used(NOW));
    // a free model through the same reach is not counted, and is not asked for its cost
    let stub = Stub::json(200, json!({"choices": [{"message": {"content": "Free."}}], "usage": {"cost": 0.5}}));
    let r = reach(&OPENROUTER, &stub).with_paid(paid("qwen/qwen3-235b-a22b", 5.0, &l));
    let c = Client::new(&r).unwrap();
    run(chat(&c, &r, "qwen/qwen3-8b:free", vec![json!({"role": "user", "content": "hi"})], 64, NOW)).unwrap();
    assert!((l.used(NOW) - 0.0008).abs() < 1e-12);
    assert!(stub.seen()[0].json().get("usage").is_none());
}

#[test]
fn at_the_cap_a_paid_call_is_refused_before_anything_is_sent() {
    let stub = Stub::json(200, json!({"choices": [{"message": {"content": "never"}}]}));
    let l = ledger();
    // the month already holds nearly all of a $5 cap; the call's worst case does not fit
    let r0 = l.reserve("mind", 4.9999, 5.0, NOW).unwrap();
    l.settle(&r0, 4.9999).unwrap();
    let r = reach(&OPENROUTER, &stub).with_paid(paid("qwen/qwen3-235b-a22b", 5.0, &l));
    let c = Client::new(&r).unwrap();
    let refused = run(chat(&c, &r, "qwen/qwen3-235b-a22b", vec![json!({"role": "user", "content": "hi"})], 1024, NOW)).unwrap_err();
    assert_eq!(refused, Failure::Capped { back_at: super::spend::next_month_start(NOW) });
    assert!(refused.sentence().contains("Monthly cap reached"));
    assert!(stub.seen().is_empty(), "nothing was sent at the cap");
}

#[test]
fn a_priced_list_keeps_open_weight_models_with_a_readable_price() {
    let list = json!({"data": [
        {"id": "qwen/qwen3-8b:free", "hugging_face_id": "Qwen/Qwen3-8B", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "qwen/qwen3-235b-a22b", "hugging_face_id": "Qwen/Qwen3-235B-A22B", "pricing": {"prompt": "0.00000013", "completion": "0.0000006"}},
        {"id": "qwen/qwen3-14b", "hugging_face_id": "Qwen/Qwen3-14B", "pricing": {"prompt": "n/a", "completion": "0.1"}},
        {"id": "closed/model", "hugging_face_id": "", "pricing": {"prompt": "0.00000125", "completion": "0.00001"}},
        {"id": "qwen/qwen2.5-vl-72b-instruct", "hugging_face_id": "Qwen/Qwen2.5-VL-72B-Instruct", "pricing": {"prompt": "0.0000007", "completion": "0.0000007"},
         "architecture": {"input_modalities": ["text", "image"]}},
    ]});
    let chat: Vec<(String, (f64, f64))> = priced(&OPENROUTER, Task::Chat, &list).into_iter().map(|(m, p)| (m.id, (p.input_per_m, p.output_per_m))).collect();
    let ids: Vec<&str> = chat.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, ["qwen/qwen2.5-vl-72b-instruct", "qwen/qwen3-235b-a22b", "qwen/qwen3-8b:free"], "a closed model and an unreadable price are left out");
    let big = &chat[1].1;
    assert!((big.0 - 0.13).abs() < 1e-9 && (big.1 - 0.6).abs() < 1e-9, "per million tokens: {big:?}");
    let vision: Vec<String> = priced(&OPENROUTER, Task::Vision, &list).into_iter().map(|(m, _)| m.id).collect();
    assert_eq!(vision, ["qwen/qwen2.5-vl-72b-instruct"]);
    // elsewhere, the free plan's models, at zero
    let groq = json!({"data": [{"id": "whisper-large-v3-turbo"}, {"id": "llama-3.3-70b-versatile"}]});
    let ears = priced(&GROQ, Task::Transcribe, &groq);
    assert_eq!(ears.len(), 1);
    assert!(ears[0].1.is_free());
    assert_eq!(reported_cost(&json!({"usage": {"cost": "0.25"}})), Some(0.25));
    assert_eq!(reported_cost(&json!({"usage": {"cost": -1}})), None);
}
