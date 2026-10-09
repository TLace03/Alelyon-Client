//! Hosted providers: the catalogue against the registry's, the model lists,
//! the registry rows, and OpenRouter's sign-in against a stub and a real
//! loopback callback. Nothing here reaches a provider or Credential Manager.

use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use futures::FutureExt;
use futures::future::BoxFuture;
use serde_json::{Value, json};

use super::openrouter::{self, Callback};
use super::*;
use crate::testkit::TempDir;

/// A request as the stub saw it: its address, its key, its JSON body.
type Seen = (String, Option<String>, Option<Value>);

/// A provider that answers every request with one reply, and records them.
#[derive(Default)]
struct Stub {
    reply: Mutex<Option<Reply>>,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Stub {
    fn answering(status: u16, body: Value) -> Stub {
        Stub {
            reply: Mutex::new(Some(Ok((status, body.to_string().into_bytes())))),
            ..Stub::default()
        }
    }
    fn next(&self) -> Reply {
        self.reply
            .lock()
            .unwrap()
            .clone()
            .unwrap_or(Err("no reply".into()))
    }
}

impl Http for Stub {
    fn get(&self, url: String, key: Option<SecretString>) -> BoxFuture<'static, Reply> {
        self.seen
            .lock()
            .unwrap()
            .push((url, key.map(|k| k.expose().to_owned()), None));
        let reply = self.next();
        async move { reply }.boxed()
    }
    fn post_json(&self, url: String, body: Value) -> BoxFuture<'static, Reply> {
        self.seen.lock().unwrap().push((url, None, Some(body)));
        let reply = self.next();
        async move { reply }.boxed()
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn p(id: &str) -> &'static Provider {
    provider(id).unwrap()
}

#[test]
fn the_providers_are_the_owners_seven_and_agree_with_the_registrys_catalogue() {
    let ids: Vec<&str> = PROVIDERS.iter().map(|p| p.id).collect();
    assert_eq!(
        ids,
        [
            "openrouter",
            "huggingface",
            "together",
            "groq",
            "fireworks",
            "deepinfra",
            "cerebras"
        ]
    );
    for provider in PROVIDERS {
        assert!(
            crate::keys::valid_key_name(provider.key_name),
            "{}",
            provider.id
        );
        assert!(provider.base_url.starts_with("https://"), "{}", provider.id);
        assert!(provider.key_page.starts_with("https://"), "{}", provider.id);
        // Where the catalogue has the provider, its address and key name are the same.
        if let Some(builtin) = registry::builtins()
            .into_iter()
            .find(|e| e.id == provider.id)
        {
            assert_eq!(builtin.base_url, provider.base_url, "{}", provider.id);
            assert_eq!(builtin.api_key_name, provider.key_name, "{}", provider.id);
            assert_eq!(builtin.kind, EndpointKind::OpenaiCompatible);
        }
    }
    assert_eq!(p("openrouter").sign_in, SignIn::OpenRouter);
    assert!(
        PROVIDERS
            .iter()
            .filter(|p| p.id != "openrouter")
            .all(|p| p.sign_in == SignIn::PasteKey)
    );
}

#[test]
fn openrouters_list_keeps_only_models_whose_weights_are_on_the_hub() {
    let list = json!({"data": [
        {"id": "openai/gpt-6", "name": "OpenAI: GPT-6", "context_length": 400000,
         "pricing": {"prompt": "0.00001", "completion": "0.00003"}},
        {"id": "meta-llama/llama-4-maverick", "name": "Meta: Llama 4 Maverick", "context_length": 1048576,
         "hugging_face_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
         "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}},
        {"id": "qwen/qwen3-235b", "name": "Qwen3 235B", "hugging_face_id": ""},
        {"id": "deepseek/deepseek-v4", "name": "DeepSeek V4", "hugging_face_id": "deepseek-ai/DeepSeek-V4"}
    ]});
    let models = parse_models(p("openrouter"), &list).unwrap();
    let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, ["deepseek/deepseek-v4", "meta-llama/llama-4-maverick"]);
    let llama = &models[1];
    assert_eq!(llama.context, Some(1_048_576));
    let (input, output) = llama.price.unwrap();
    assert!((input - 0.15).abs() < 1e-9 && (output - 0.6).abs() < 1e-9);
    assert_eq!(models[0].price, None);
}

#[test]
fn other_lists_leave_out_what_is_not_a_chat_model() {
    // Together answers a bare list and says each row's type.
    let together = json!([
        {"id": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "display_name": "Llama 3.3 70B", "type": "chat", "context_length": 131072},
        {"id": "black-forest-labs/FLUX.1-schnell", "type": "image"},
        {"id": "BAAI/bge-large-en-v1.5", "type": "embedding"}
    ]);
    let models = parse_models(p("together"), &together).unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "Llama 3.3 70B");
    assert_eq!(models[0].context, Some(131_072));
    // Groq says no type: speech models are left out by their id.
    let groq = json!({"data": [
        {"id": "llama-3.3-70b-versatile", "context_window": 131072},
        {"id": "whisper-large-v3"},
        {"id": "playai-tts"},
        {"id": "llama-3.3-70b-versatile"}
    ]});
    let models = parse_models(p("groq"), &groq).unwrap();
    assert_eq!(
        models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        ["llama-3.3-70b-versatile"]
    );
    assert_eq!(models[0].context, Some(131_072));
    assert_eq!(parse_models(p("groq"), &json!({"models": []})), None);
}

#[test]
fn a_list_is_read_with_the_key_and_a_refusal_is_said_in_words() {
    let rt = runtime();
    let stub = Stub::answering(200, json!({"data": [{"id": "Qwen/Qwen3-8B"}]}));
    let models = rt
        .block_on(list_models(
            &stub,
            p("huggingface"),
            Some(SecretString::new("hf_x")),
        ))
        .unwrap();
    assert_eq!(models[0].id, "Qwen/Qwen3-8B");
    let seen = stub.seen.lock().unwrap().clone();
    assert_eq!(seen[0].0, "https://router.huggingface.co/v1/models");
    assert_eq!(seen[0].1.as_deref(), Some("hf_x"));

    // A provider that lists only with a key is not asked without one.
    let stub = Stub::answering(200, json!({"data": []}));
    let err = rt
        .block_on(list_models(&stub, p("groq"), None))
        .unwrap_err();
    assert_eq!(err, "Connect Groq first: its model list needs your key.");
    assert!(stub.seen.lock().unwrap().is_empty());

    let stub = Stub::answering(401, json!({"error": "bad key sk-secret"}));
    let err = rt
        .block_on(list_models(
            &stub,
            p("together"),
            Some(SecretString::new("sk-secret")),
        ))
        .unwrap_err();
    assert_eq!(
        err,
        "Together AI did not accept the key: make a new one and connect again."
    );
    assert!(!err.contains("sk-secret"));
}

#[test]
fn connecting_and_choosing_write_the_providers_registry_row() {
    let dir = TempDir::new("hosted-registry");
    let path = dir.path().join("model_endpoints.json");
    // OpenRouter is in the catalogue: its built-in row is enabled.
    enable(&path, p("openrouter")).unwrap();
    choose(&path, p("openrouter"), " meta-llama/llama-4-maverick ").unwrap();
    // DeepInfra is not: a row of the reader's own is written.
    enable(&path, p("deepinfra")).unwrap();
    let rows = registry::load_with_report(&path).endpoints;
    let openrouter = rows.iter().find(|e| e.id == "openrouter").unwrap();
    assert!(openrouter.enabled && openrouter.builtin);
    assert_eq!(openrouter.model, "meta-llama/llama-4-maverick");
    assert_eq!(openrouter.api_key_name, "OPENROUTER_API_KEY");
    let deepinfra = rows.iter().find(|e| e.id == "deepinfra").unwrap();
    assert!(deepinfra.enabled && !deepinfra.builtin);
    assert_eq!(deepinfra.base_url, "https://api.deepinfra.com/v1/openai");
    assert_eq!(deepinfra.api_key_name, "DEEPINFRA_API_KEY");
    assert_eq!(deepinfra.kind, EndpointKind::OpenaiCompatible);
    // Choosing keeps what connecting wrote.
    choose(&path, p("deepinfra"), "deepseek-ai/DeepSeek-V4").unwrap();
    let rows = registry::load_with_report(&path).endpoints;
    assert_eq!(rows.iter().filter(|e| e.id == "deepinfra").count(), 1);
    assert_eq!(
        choose(&path, p("deepinfra"), "  "),
        Err("Choose a model first.".to_owned())
    );

    // A file that cannot be read whole is not changed.
    std::fs::write(&path, b"{not json").unwrap();
    assert!(enable(&path, p("groq")).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"{not json");
}

#[test]
fn the_challenge_is_rfc_7636s_and_the_page_carries_it() {
    // RFC 7636 appendix B.
    assert_eq!(
        openrouter::challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    );
    assert_eq!(openrouter::base64url(b"f"), "Zg");
    assert_eq!(openrouter::base64url(b"fo"), "Zm8");
    assert_eq!(openrouter::base64url(b"foo"), "Zm9v");
    let url = url::Url::parse(&openrouter::auth_url(51423, "CH", "ST")).unwrap();
    assert_eq!(url.host_str(), Some("openrouter.ai"));
    let pairs: Vec<(String, String)> = url.query_pairs().into_owned().collect();
    let get = |k: &str| pairs.iter().find(|(n, _)| n == k).unwrap().1.clone();
    assert_eq!(get("callback_url"), "http://localhost:51423/callback");
    assert_eq!(get("code_challenge"), "CH");
    assert_eq!(get("code_challenge_method"), "S256");
    assert_eq!(get("state"), "ST");
    assert_eq!(get("key_label"), openrouter::KEY_LABEL);
}

#[test]
fn only_our_callback_with_its_state_is_a_code() {
    let state = "abc";
    assert_eq!(
        openrouter::parse_callback(
            "GET /callback?code=K1&state=abc HTTP/1.1\r\nHost: x\r\n\r\n",
            state
        ),
        Callback::Code("K1".into())
    );
    assert_eq!(
        openrouter::parse_callback("GET /callback?code=K1&state=zzz HTTP/1.1\r\n", state),
        Callback::Other
    );
    assert_eq!(
        openrouter::parse_callback("GET /callback?code=K1 HTTP/1.1\r\n", state),
        Callback::Other
    );
    assert_eq!(
        openrouter::parse_callback("GET /favicon.ico HTTP/1.1\r\n", state),
        Callback::Other
    );
    assert_eq!(
        openrouter::parse_callback("POST /callback?code=K1&state=abc HTTP/1.1\r\n", state),
        Callback::Other
    );
    assert_eq!(
        openrouter::parse_callback("GET /callback?error=access_denied HTTP/1.1\r\n", state),
        Callback::Refused
    );
}

#[test]
fn the_code_is_traded_for_the_key_with_the_verifier() {
    let rt = runtime();
    let stub = Stub::answering(200, json!({"key": "sk-or-v1-abc"}));
    let key = rt
        .block_on(openrouter::exchange(&stub, "CODE", "VERIFIER"))
        .unwrap();
    assert_eq!(key.expose(), "sk-or-v1-abc");
    let seen = stub.seen.lock().unwrap().clone();
    assert_eq!(seen[0].0, openrouter::KEYS_URL);
    assert_eq!(
        seen[0].2,
        Some(json!({"code": "CODE", "code_verifier": "VERIFIER", "code_challenge_method": "S256"}))
    );
    let stub = Stub::answering(403, json!({"error": "expired"}));
    let err = rt
        .block_on(openrouter::exchange(&stub, "CODE", "V"))
        .unwrap_err();
    assert_eq!(
        err,
        "OpenRouter refused the sign-in (it lasts 10 minutes): try again."
    );
    let stub = Stub::answering(200, json!({"nokey": true}));
    assert_eq!(
        rt.block_on(openrouter::exchange(&stub, "CODE", "V"))
            .unwrap_err(),
        "OpenRouter's answer held no key."
    );
}

/// Ask the callback listener as a browser would, and read its page.
fn browse(port: u16, target: &str) -> String {
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(
        stream,
        "GET {target} HTTP/1.1\r\nHost: localhost:{port}\r\n\r\n"
    )
    .unwrap();
    let mut page = String::new();
    stream.read_to_string(&mut page).unwrap();
    page
}

#[test]
fn a_sign_in_waits_for_its_own_callback_and_returns_the_key() {
    let rt = runtime();
    let pending = rt.block_on(openrouter::begin()).unwrap();
    let url = url::Url::parse(pending.url()).unwrap();
    let pairs: Vec<(String, String)> = url.query_pairs().into_owned().collect();
    let get = |k: &str| pairs.iter().find(|(n, _)| n == k).unwrap().1.clone();
    let callback = url::Url::parse(&get("callback_url")).unwrap();
    assert_eq!(callback.host_str(), Some("localhost"));
    let port = callback.port().unwrap();
    let state = get("state");
    let challenge = get("code_challenge");

    let stub = Arc::new(Stub::answering(200, json!({"key": "sk-or-v1-signed"})));
    let finish = {
        let stub = stub.clone();
        rt.spawn(async move { pending.finish(stub.as_ref()).await })
    };
    // A stray request and a forged state are answered and ignored.
    assert!(browse(port, "/favicon.ico").starts_with("HTTP/1.1 404"));
    assert!(browse(port, "/callback?code=EVIL&state=wrong").starts_with("HTTP/1.1 404"));
    let page = browse(port, &format!("/callback?code=GOOD&state={state}"));
    assert!(page.starts_with("HTTP/1.1 200") && page.contains("Signed in to OpenRouter"));
    let key = rt.block_on(finish).unwrap().unwrap();
    assert_eq!(key.expose(), "sk-or-v1-signed");
    // The code sent is the callback's, with the verifier whose challenge the page carried.
    let seen = stub.seen.lock().unwrap().clone();
    let body = seen[0].2.clone().unwrap();
    assert_eq!(body["code"], "GOOD");
    assert_eq!(
        openrouter::challenge(body["code_verifier"].as_str().unwrap()),
        challenge
    );
}

#[test]
fn a_refused_sign_in_says_so() {
    let rt = runtime();
    let pending = rt.block_on(openrouter::begin()).unwrap();
    let url = url::Url::parse(pending.url()).unwrap();
    let callback = url
        .query_pairs()
        .find(|(k, _)| k == "callback_url")
        .map(|(_, v)| url::Url::parse(&v).unwrap())
        .unwrap();
    let stub = Arc::new(Stub::default());
    let finish = {
        let stub = stub.clone();
        rt.spawn(async move { pending.finish(stub.as_ref()).await })
    };
    let page = browse(callback.port().unwrap(), "/callback?error=access_denied");
    assert!(page.contains("OpenRouter was not connected"));
    assert_eq!(
        rt.block_on(finish).unwrap().unwrap_err(),
        "OpenRouter was not connected: the key was not approved."
    );
    assert!(stub.seen.lock().unwrap().is_empty());
}

/// By hand (`--ignored`): the two lists readable without a key, from the real
/// providers. No key is sent and nothing is kept.
#[test]
#[ignore = "reaches openrouter.ai and router.huggingface.co"]
fn the_real_keyless_lists_hold_open_weight_chat_models() {
    let rt = runtime();
    let web = Web::new().unwrap();
    for id in ["openrouter", "huggingface"] {
        let models = rt.block_on(list_models(&web, p(id), None)).unwrap();
        eprintln!(
            "{id}: {} open-weight chat models; first {:?}",
            models.len(),
            models.iter().take(3).map(|m| &m.id).collect::<Vec<_>>()
        );
        assert!(models.len() >= 10, "{id}: {}", models.len());
    }
}
