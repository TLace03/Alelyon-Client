use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::FutureExt;
use futures::future::BoxFuture;
use serde_json::{Value, json};

use super::catalog::{self, Tier};
use super::fim::{self, Family};
use super::settings::{self, Choice, Settings};
use super::*;
use crate::env::MapEnv;
use crate::llama::files::{LlamaPaths, LocalModel};
use crate::llama::{Lease, LlamaError, ManagedEndpoint, Opened};
use crate::testkit::TempDir;

type Seen = (String, Option<String>, Option<Value>);

#[derive(Default)]
struct Stub {
    replies: Mutex<VecDeque<Reply>>,
    seen: Mutex<Vec<Seen>>,
}

impl Stub {
    fn answering(replies: Vec<Reply>) -> Arc<Stub> {
        Arc::new(Stub { replies: Mutex::new(replies.into()), ..Stub::default() })
    }
    fn next(&self) -> Reply {
        self.replies.lock().unwrap().pop_front().unwrap_or(Err("no reply".into()))
    }
    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

impl Wire for Stub {
    fn get(&self, url: String, key: Option<SecretString>) -> BoxFuture<'static, Reply> {
        self.seen.lock().unwrap().push((url, key.map(|k| k.expose().to_owned()), None));
        let reply = self.next();
        async move { reply }.boxed()
    }
    fn post(&self, url: String, key: SecretString, body: Value) -> BoxFuture<'static, Reply> {
        self.seen.lock().unwrap().push((url, Some(key.expose().to_owned()), Some(body)));
        let reply = self.next();
        async move { reply }.boxed()
    }
}

/// A runtime that runs `running` (if any) and opens only through the
/// trait's own unswitched check.
struct Runtime {
    running: Option<String>,
    opens: AtomicUsize,
}

impl ManagedRuntime for Runtime {
    fn running(&self) -> Option<String> {
        self.running.clone()
    }
    fn failed(&self) -> Option<String> {
        None
    }
    fn open(&self, model: LocalModel) -> BoxFuture<'static, Result<Opened, LlamaError>> {
        self.opens.fetch_add(1, Ordering::Relaxed);
        async move {
            Ok(Opened {
                endpoint: ManagedEndpoint {
                    base_url: "http://127.0.0.1:4711".into(),
                    alias: model.name,
                    token: SecretString::new("launch-token"),
                    binary_sha256: None,
                    generation: 1,
                },
                lease: Lease::detached(),
            })
        }
        .boxed()
    }
}

struct Fixture {
    dir: TempDir,
    completer: Completer,
    runtime: Arc<Runtime>,
}

fn fixture(tag: &str, keys: &[(&str, &str)], running: Option<&str>, wire: Arc<Stub>) -> Fixture {
    let dir = TempDir::new(tag);
    let home = dir.path().join("home");
    let mut env = MapEnv::new().with("USERPROFILE", home.as_os_str()).with("HOME", home.as_os_str());
    for (name, value) in keys {
        env.set(name, *value);
    }
    let paths = LlamaPaths::from_env(&env);
    assert!(paths.models_dir.starts_with(dir.path()), "the models folder is the test's own");
    std::fs::create_dir_all(&paths.models_dir).unwrap();
    crate::llama::files::tests::gguf(&paths.models_dir.join("qwen2.5-coder-1.5b.gguf"));
    crate::llama::files::tests::gguf(&paths.models_dir.join("chat-model.gguf"));
    let runtime = Arc::new(Runtime { running: running.map(str::to_owned), opens: AtomicUsize::new(0) });
    let completer =
        Completer::new(Arc::new(env), StateRoot::at(dir.path().join("state")), runtime.clone(), wire);
    Fixture { dir, completer, runtime }
}

fn block<T>(future: impl std::future::Future<Output = T>) -> T {
    futures::executor::block_on(future)
}

fn request(prefix: &str, suffix: &str) -> Request {
    Request { prefix: prefix.into(), suffix: suffix.into() }
}

#[test]
fn families_are_known_by_their_names_and_ask_in_their_own_tokens() {
    assert_eq!(fim::family("qwen/qwen-2.5-coder-32b-instruct"), Some(Family::Qwen));
    assert_eq!(fim::family("Qwen3-Coder-30B-A3B"), Some(Family::Qwen));
    assert_eq!(fim::family("mistralai/codestral-2508"), Some(Family::Codestral));
    assert_eq!(fim::family("deepseek-ai/deepseek-coder-6.7b-base"), Some(Family::DeepSeek));
    assert_eq!(fim::family("bigcode/starcoder2-15b"), Some(Family::StarCoder));
    assert_eq!(fim::family("ibm-granite/granite-8b-code-base"), Some(Family::StarCoder));
    assert_eq!(fim::family("google/codegemma-7b"), Some(Family::CodeGemma));
    assert_eq!(fim::family("meta/codellama-34b"), Some(Family::CodeLlama));
    for not in ["qwen/qwen3-8b", "meta-llama/llama-3.3-70b", "deepseek/deepseek-chat", ""] {
        assert_eq!(fim::family(not), None, "{not}");
    }
    assert_eq!(Family::Qwen.prompt("a", "b"), "<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>");
    assert_eq!(Family::Codestral.prompt("a", "b"), "[SUFFIX]b[PREFIX]a");
    assert_eq!(Family::StarCoder.prompt("a", "b"), "<fim_prefix>a<fim_suffix>b<fim_middle>");
}

#[test]
fn a_request_sends_whole_lines_within_its_limits() {
    let short = request("fn main() {\n    let x", " = 1;\n}\n");
    assert_eq!(short.clipped(), ("fn main() {\n    let x", " = 1;\n}\n"));

    let line = format!("{}\n", "a".repeat(99));
    let long = request(&format!("{}tail", line.repeat(100)), &format!("head{}", line.repeat(100)));
    let (prefix, suffix) = long.clipped();
    assert!(prefix.chars().count() <= PREFIX_CHARS && suffix.chars().count() <= SUFFIX_CHARS);
    assert!(prefix.starts_with('a') && prefix.ends_with("tail"), "the prefix starts at a line's start");
    assert!(suffix.starts_with("head") && suffix.ends_with('\n'), "the suffix ends at a line's end");

    // Characters, not bytes: a clip never splits one.
    let wide = request(&"é".repeat(PREFIX_CHARS + 5), "");
    assert_eq!(wide.clipped().0.chars().count(), PREFIX_CHARS);
}

#[test]
fn an_answer_is_cut_where_it_stops_or_repeats_what_follows() {
    assert_eq!(clean("a + b<|endoftext|>junk", ""), Some("a + b".into()));
    assert_eq!(clean("   \n  ", ""), None);
    // The model writes the closing line that is already after the cursor.
    assert_eq!(
        clean("x + 1\n    }\n    more", "\n    }\n"),
        Some("x + 1".into()),
        "cut before the repeated line"
    );
    // The rest of the cursor's line, written again at the end.
    assert_eq!(clean("a, b)", ")"), Some("a, b".into()));
    let many: String = (0..30).map(|i| format!("line{i}\n")).collect();
    assert_eq!(clean(&many, "").unwrap().lines().count(), MAX_LINES);
}

#[test]
fn costs_are_per_thousand_completions_and_grouped() {
    // 2,000 tokens in and 64 out, a thousand times.
    assert!((catalog::per_thousand((1.0, 2.0)) - (2.0 + 0.128)).abs() < 1e-9);
    assert_eq!(catalog::tier(Some(0.0)), Tier::Free);
    assert_eq!(catalog::tier(Some(0.2)), Tier::UnderQuarter);
    assert_eq!(catalog::tier(Some(0.25)), Tier::UnderDollar);
    assert_eq!(catalog::tier(Some(1.5)), Tier::OverDollar);
    assert_eq!(catalog::tier(None), Tier::Unlisted);
    assert_eq!(catalog::cost_text(Some(0.0)), "free");
    assert_eq!(catalog::cost_text(Some(0.1234)), "about $0.123 per 1,000");
}

#[test]
fn the_catalogue_lists_local_files_and_providers_fill_in_the_middle_models_by_price() {
    let openrouter = json!({"data": [
        {"id": "qwen/qwen-2.5-coder-32b-instruct", "name": "Qwen2.5 Coder 32B",
         "hugging_face_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
         "pricing": {"prompt": "0.00000006", "completion": "0.00000015"}},
        {"id": "mistralai/codestral-2508", "name": "Codestral",
         "pricing": {"prompt": "0.0000003", "completion": "0.0000009"}},
        {"id": "qwen/qwen-2.5-coder-32b-instruct:free", "name": "Qwen2.5 Coder 32B (free)",
         "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B",
         "hugging_face_id": "meta-llama/Llama-3.3-70B-Instruct",
         "pricing": {"prompt": "0.0000001", "completion": "0.0000003"}}
    ]});
    let together = json!([
        {"id": "Qwen/Qwen2.5-Coder-32B-Instruct", "type": "chat", "pricing": {"input": 0.8, "output": 0.8}},
        {"id": "deepseek-ai/DeepSeek-V3", "type": "chat", "pricing": {"input": 1.25, "output": 1.25}}
    ]);
    let stub = Stub::answering(vec![Ok((200, openrouter.to_string().into_bytes())), Ok((200, together.to_string().into_bytes()))]);
    let f = fixture("complete-catalog", &[("TOGETHER_API_KEY", "together-key")], None, stub.clone());
    let (offers, notes) = block(f.completer.offers());
    assert!(notes.is_empty(), "{notes:?}");
    let urls: Vec<_> = stub.seen().into_iter().map(|s| (s.0, s.1)).collect();
    assert_eq!(
        urls,
        vec![
            ("https://openrouter.ai/api/v1/models".into(), None),
            ("https://api.together.xyz/v1/models".into(), Some("together-key".into())),
        ],
        "OpenRouter lists without a key; Together with the reader's; providers with no key are not asked"
    );
    let names: Vec<(&str, &str, Tier, bool)> =
        offers.iter().map(|o| (o.name.as_str(), o.host.as_str(), o.tier, o.ready)).collect();
    assert!(names.contains(&("chat-model", "This PC", Tier::Free, true)));
    assert!(names.contains(&("qwen2.5-coder-1.5b", "This PC", Tier::Free, true)));
    assert!(names.contains(&("Codestral", "OpenRouter", Tier::UnderDollar, false)), "not open-weight, still offered");
    assert!(names.contains(&("Qwen2.5 Coder 32B", "OpenRouter", Tier::UnderQuarter, false)));
    assert!(names.contains(&("Qwen2.5 Coder 32B (free)", "OpenRouter", Tier::Free, false)));
    assert!(names.contains(&("Qwen/Qwen2.5-Coder-32B-Instruct", "Together AI", Tier::OverDollar, true)));
    assert!(!names.iter().any(|n| n.0.contains("Llama") || n.0.contains("DeepSeek-V3")), "chat-only models are left out");
    let local = offers.iter().find(|o| o.name == "qwen2.5-coder-1.5b").unwrap();
    assert!(local.fim && !offers.iter().find(|o| o.name == "chat-model").unwrap().fim);

    let groups = catalog::grouped(offers);
    let tiers: Vec<Tier> = groups.iter().map(|g| g.0).collect();
    assert_eq!(tiers, vec![Tier::Free, Tier::UnderQuarter, Tier::UnderDollar, Tier::OverDollar]);
    assert_eq!(groups[0].1[0].name, "qwen2.5-coder-1.5b", "in a group, a model usable now first");
}

#[test]
fn settings_are_kept_whole_and_default_to_off() {
    let dir = TempDir::new("complete-settings");
    let path = dir.path().join("lattice_native").join("completion.json");
    assert_eq!(settings::load(&path), Settings::default());
    let chosen = Settings {
        on: true,
        choice: Some(Choice::Hosted { provider: "openrouter".into(), model: "mistralai/codestral-2508".into() }),
    };
    settings::save(&path, &chosen).unwrap();
    assert_eq!(settings::load(&path), chosen);
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("\"source\": \"hosted\""), "{text}");
    std::fs::write(&path, "not json").unwrap();
    assert_eq!(settings::load(&path), Settings::default(), "unreadable means off");
}

#[test]
fn a_local_completion_asks_the_servers_infill_with_its_token() {
    let stub = Stub::answering(vec![Ok((200, json!({"content": "1 + 2;\n}\n<|endoftext|>"}).to_string().into_bytes()))]);
    let f = fixture("complete-local", &[], None, stub.clone());
    let choice = Choice::Local { model: "qwen2.5-coder-1.5b".into() };
    let answer = block(f.completer.complete(&choice, &request("let x = ", "\n}\n")));
    assert_eq!(answer, Ok(Some("1 + 2;".into())), "cut before the repeated closing line");
    let seen = stub.seen();
    assert_eq!(seen[0].0, "http://127.0.0.1:4711/infill");
    assert_eq!(seen[0].1.as_deref(), Some("launch-token"));
    let body = seen[0].2.clone().unwrap();
    assert_eq!(body["input_prefix"], "let x = ");
    assert_eq!(body["input_suffix"], "\n}\n");
    assert_eq!(body["n_predict"], MAX_TOKENS);
    assert!(f.dir.path().exists());
}

#[test]
fn a_local_completion_never_switches_the_servers_model() {
    let stub = Stub::answering(vec![]);
    let f = fixture("complete-serving", &[], Some("chat-model"), stub.clone());
    let choice = Choice::Local { model: "qwen2.5-coder-1.5b".into() };
    let answer = block(f.completer.complete(&choice, &request("a", "")));
    assert_eq!(answer, Err(LlamaError::Serving.sentence().to_owned()));
    assert_eq!(f.runtime.opens.load(Ordering::Relaxed), 0, "nothing was opened");
    assert!(stub.seen().is_empty(), "nothing was sent");

    let missing = block(f.completer.complete(&Choice::Local { model: "nope".into() }, &request("a", "")));
    assert_eq!(missing, Err("nope is not in the models folder.".to_owned()));
}

#[test]
fn a_hosted_completion_asks_the_providers_completions_in_the_familys_tokens() {
    let stub = Stub::answering(vec![
        Ok((200, json!({"choices": [{"text": "b)"}]}).to_string().into_bytes())),
        Ok((402, Vec::new())),
    ]);
    let f = fixture("complete-hosted", &[("OPENROUTER_API_KEY", "reader-key")], None, stub.clone());
    let choice = Choice::Hosted { provider: "openrouter".into(), model: "qwen/qwen-2.5-coder-32b-instruct".into() };
    let answer = block(f.completer.complete(&choice, &request("f(a, ", ")\n")));
    assert_eq!(answer, Ok(Some("b".into())), "the rest of the cursor's line is not written twice");
    let seen = stub.seen();
    assert_eq!(seen[0].0, "https://openrouter.ai/api/v1/completions");
    assert_eq!(seen[0].1.as_deref(), Some("reader-key"));
    let body = seen[0].2.clone().unwrap();
    assert_eq!(body["prompt"], "<|fim_prefix|>f(a, <|fim_suffix|>)\n<|fim_middle|>");
    assert_eq!(body["model"], "qwen/qwen-2.5-coder-32b-instruct");
    assert!(body["stop"].as_array().unwrap().len() <= 4, "providers take at most four stops");

    let broke = block(f.completer.complete(&choice, &request("x", "")));
    assert_eq!(broke, Err("OpenRouter says the account has no credit left.".to_owned()));
}

#[test]
fn a_hosted_completion_sends_nothing_that_looks_like_a_key_or_without_one() {
    let stub = Stub::answering(vec![]);
    let f = fixture("complete-refused", &[("OPENROUTER_API_KEY", "reader-key")], None, stub.clone());
    let choice = Choice::Hosted { provider: "openrouter".into(), model: "mistralai/codestral-2508".into() };
    let secret = format!("API_KEY = \"sk-ant-api03-{}\"\n", "a".repeat(80));
    let refused = block(f.completer.complete(&choice, &request(&secret, "")));
    assert!(refused.unwrap_err().contains("looks like it holds a key"));

    let chat_only = Choice::Hosted { provider: "openrouter".into(), model: "meta-llama/llama-3.3-70b-instruct".into() };
    assert!(block(f.completer.complete(&chat_only, &request("a", ""))).unwrap_err().contains("not a fill-in-the-middle"));
    let groq = Choice::Hosted { provider: "groq".into(), model: "qwen-2.5-coder-32b".into() };
    assert_eq!(block(f.completer.complete(&groq, &request("a", ""))), Err("That provider does not offer completions.".to_owned()));

    let g = fixture("complete-nokey", &[], None, stub.clone());
    assert_eq!(block(g.completer.complete(&choice, &request("a", ""))), Err("Connect OpenRouter first.".to_owned()));
    assert!(stub.seen().is_empty(), "nothing was sent");
}
